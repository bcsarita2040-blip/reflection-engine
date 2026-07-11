import streamlit as st
import asyncio
import aiohttp
import nest_asyncio
import time
import random

# Apply the nest_asyncio monkey patch (Created by Ewald de Wit) 
nest_asyncio.apply()

st.set_page_config(
    page_title="Multi-LLM Cross-Reflection Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# The definitive list of current models.
AVAILABLE_MODELS = {
    "Meta: Llama 3.3 70B Instruct (Free)": "meta-llama/llama-3.3-70b-instruct:free",
    "Google DeepMind: Gemini 2.5 Flash (Free)": "google/gemini-2.5-flash:free",
    "Google DeepMind: Gemini 3.5 Flash (Free)": "google/gemini-3.5-flash:free",
    "Google DeepMind: Gemini 3.1 Pro Preview (Free)": "google/gemini-3.1-pro-preview:free",
    "OpenAI: ChatGPT-4o Mini": "openai/gpt-4o-mini",
    "Anthropic: Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "NVIDIA: Nemotron 3 Ultra 550B (Free)": "nvidia/nemotron-3-ultra:free",
    "Tencent: Hy3 MoE 295B (Free)": "tencent/hy3:free",
    "Cohere: North Mini Code (Free)": "cohere/north-mini-code:free"
}

if "worker_responses" not in st.session_state:
    st.session_state.worker_responses = {}
if "final_synthesis" not in st.session_state:
    st.session_state.final_synthesis = ""
if "execution_time" not in st.session_state:
    st.session_state.execution_time = 0.0

# -----------------------------------------------------------------------------
# ASYNCHRONOUS API ORCHESTRATION ENGINE (WITH FALLBACKS)
# -----------------------------------------------------------------------------

async def fetch_llm_response(
    session: aiohttp.ClientSession, 
    model_friendly_name: str, 
    model_id: str, 
    api_key: str, 
    messages: list,
    thinking_mode: bool = False,
    max_retries: int = 5,
    initial_delay: float = 2.0,
    max_delay: float = 16.0,
    backoff_factor: float = 2.0
) -> dict:
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-OpenRouter-Title": "Multi-LLM Cross-Reflection Engine"
    }
    
    # THE FALLBACK UPGRADE: We pass an array of models. 
    # If the first one chokes, it instantly routes to a generic free model.
    payload = {
        "models": [model_id, "openrouter/free"], 
        "messages": messages,
        "temperature": 0.5 
    }
    
    if thinking_mode:
        payload["reasoning"] = {"effort": "high"}

    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, json=payload, headers=headers, timeout=60) as response:
                status = response.status
                
                if status == 200:
                    data = await response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        
                        reasoning_content = data["choices"][0]["message"].get("reasoning_details", "")
                        if reasoning_content:
                            content = f"### 🧠 Internal Thoughts:\n_{reasoning_content}_\n\n---\n\n### 🗣️ Final Answer:\n{content}"

                        # Check if OpenRouter had to use the backup model
                        actual_model_used = data.get("model", model_id)
                        display_name = model_friendly_name
                        if actual_model_used != model_id and "openrouter" not in actual_model_used:
                            display_name += f" (⚠️ Backup Triggered: {actual_model_used})"

                        return {
                            "friendly_name": display_name,
                            "model_id": actual_model_used,
                            "success": True,
                            "text": content,
                            "error": None
                        }
                    else:
                        return {
                            "friendly_name": model_friendly_name,
                            "model_id": model_id,
                            "success": False,
                            "text": "",
                            "error": "API returned empty data."
                        }
                
                elif status in [429, 500, 502, 503, 504]:
                    if attempt == max_retries:
                        return {
                            "friendly_name": model_friendly_name,
                            "model_id": model_id,
                            "success": False,
                            "text": "",
                            "error": f"Failed after {max_retries} attempts (Status {status})."
                        }
                else:
                    try:
                        err_payload = await response.json()
                        err_msg = err_payload.get("error", {}).get("message", "No message provided")
                    except Exception:
                        err_msg = "Unknown error."
                    return {
                        "friendly_name": model_friendly_name,
                        "model_id": model_id,
                        "success": False,
                        "text": "",
                        "error": f"HTTP {status} - {err_msg}"
                    }
                    
        except asyncio.TimeoutError:
            if attempt == max_retries:
                return {
                    "friendly_name": model_friendly_name,
                    "model_id": model_id,
                    "success": False,
                    "text": "",
                    "error": "Request timed out."
                }
        except Exception as e:
            if attempt == max_retries:
                return {
                    "friendly_name": model_friendly_name,
                    "model_id": model_id,
                    "success": False,
                    "text": "",
                    "error": f"Error: {str(e)}"
                }
        
        delay = min(max_delay, initial_delay * (backoff_factor ** (attempt - 1)))
        jitter = random.uniform(0.0, 0.1 * delay)
        await asyncio.sleep(delay + jitter)

    return {
        "friendly_name": model_friendly_name,
        "model_id": model_id,
        "success": False,
        "text": "",
        "error": "Failed completely."
    }

async def execute_parallel_run(selected_workers: dict, api_key: str, prompt: list, thinking_mode: bool) -> list:
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_llm_response(session, friendly_name, model_id, api_key, prompt, thinking_mode)
            for friendly_name, model_id in selected_workers.items()
        ]
        return await asyncio.gather(*tasks)

# -----------------------------------------------------------------------------
# INTERFACE
# -----------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=70)
    st.title("Engine Config")
    openrouter_key = st.text_input("OpenRouter API Key", type="password")
    
    st.markdown("### Model Configuration")
    worker_selections = st.multiselect(
        "Choose Phase 1 Workers",
        options=list(AVAILABLE_MODELS.keys()),
        default=[
            "Google DeepMind: Gemini 3.5 Flash (Free)",
            "Google DeepMind: Gemini 3.1 Pro Preview (Free)",
            "Meta: Llama 3.3 70B Instruct (Free)"
        ]
    )
    
    judge_selection = st.selectbox(
        "Choose Phase 2 Judge",
        options=list(AVAILABLE_MODELS.keys()),
        index=4
    )
    
    thinking_mode = st.toggle("🧠 Enable Thinking Mode", value=False)

st.title("🤖 Multi-LLM Cross-Reflection Engine")
st.caption("Now with automated fallback routing!")

user_prompt = st.text_area("Submit your prompt:", height=120)

if st.button("Initialize Engine", type="primary"):
    if not openrouter_key:
        st.error("Missing OpenRouter API key.")
    elif not worker_selections:
        st.error("Select at least one Worker model.")
    elif not user_prompt.strip():
        st.warning("Input a valid prompt.")
    else:
        active_workers = {name: AVAILABLE_MODELS[name] for name in worker_selections}
        judge_model_id = AVAILABLE_MODELS[judge_selection]
        formatted_messages = [{"role": "user", "content": user_prompt}]
        
        st.markdown("### Phase 1: Simultaneous Execution")
        progress_tracker = st.empty()
        progress_tracker.info("Dispatching requests in parallel (with backups ready)...")
        
        start_timestamp = time.time()
        loop = asyncio.get_event_loop()
        worker_outputs = loop.run_until_complete(
            execute_parallel_run(active_workers, openrouter_key, formatted_messages, thinking_mode)
        )
        
        progress_tracker.empty()
        
        columns = st.columns(len(worker_outputs), border=True)
        for idx, col in enumerate(columns):
            response_data = worker_outputs[idx]
            with col:
                st.markdown(f"##### 🟢 {response_data['friendly_name']}")
                if response_data["success"]:
                    st.write(response_data["text"])
                else:
                    st.error(response_data["error"])
        
        st.session_state.worker_responses = {
            item["friendly_name"]: item["text"] for item in worker_outputs if item["success"]
        }
        
        if st.session_state.worker_responses:
            st.markdown("---")
            st.markdown("### Phase 2: Cross-Analysis & Synthesis")
            judge_loader = st.empty()
            judge_loader.info("Synthesizing final consensus...")
            
            synthesis_input = f"Original Query: {user_prompt}\n\n"
            for source, raw in st.session_state.worker_responses.items():
                synthesis_input += f"=== Raw Response [{source}] ===\n{raw}\n\n"
            
            judge_messages = [
                {"role": "system", "content": "You are an elite research consensus arbitrator. Expose flaws and compile a flawless master response."},
                {"role": "user", "content": synthesis_input}
            ]
            
            async def run_judge():
                async with aiohttp.ClientSession() as session:
                    return await fetch_llm_response(
                        session, judge_selection, judge_model_id, openrouter_key, judge_messages, thinking_mode
                    )
            
            judge_response = loop.run_until_complete(run_judge())
            st.session_state.execution_time = time.time() - start_timestamp
            judge_loader.empty()
            
            if judge_response["success"]:
                st.session_state.final_synthesis = judge_response["text"]
                st.success(f"Synthesis compiled successfully in {st.session_state.execution_time:.2f} seconds.")
                st.markdown(st.session_state.final_synthesis)
            else:
                st.error(f"Synthesis failed: {judge_response['error']}")

elif st.session_state.final_synthesis:
    st.markdown("---")
    st.markdown("### Phase 1: Simultaneous Execution (Cached)")
    columns = st.columns(len(st.session_state.worker_responses), border=True)
    for idx, (header, text) in enumerate(st.session_state.worker_responses.items()):
        with columns[idx]:
            st.markdown(f"##### 🟢 {header}")
            st.write(text)
            
    st.markdown("---")
    st.markdown("### Phase 2: Cross-Analysis & Synthesis (Cached)")
    st.markdown(st.session_state.final_synthesis)
