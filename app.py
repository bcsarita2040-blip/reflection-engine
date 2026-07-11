import streamlit as st
import asyncio
import aiohttp
import nest_asyncio
import time
import random

# Apply the nest_asyncio monkey patch to allow running nested event loops
# within Streamlit's pre-existing Tornado web server environment.
nest_asyncio.apply()

# Configure the visual layout of the Streamlit interface to wide mode.
# Wide mode is critical for side-by-side rendering of multiple worker LLM responses.
st.set_page_config(
    page_title="Multi-LLM Cross-Reflection Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the dictionary of available free-tier models on OpenRouter as of mid-2026.
# These identifiers map directly to OpenRouter's chat completion endpoints.
AVAILABLE_MODELS = {
    "Meta: Llama 3.3 70B Instruct (Free)": "meta-llama/llama-3.3-70b-instruct:free",
    "Google: Gemini 2.5 Flash (Free)": "google/gemini-2.5-flash:free",
    "NVIDIA: Nemotron 3 Ultra 550B (Free)": "nvidia/nemotron-3-ultra:free",
    "NVIDIA: Nemotron 3 Super 120B (Free)": "nvidia/nemotron-3-super:free",
    "Tencent: Hy3 MoE 295B (Free)": "tencent/hy3:free",
    "Cohere: North Mini Code (Free)": "cohere/north-mini-code:free",
    "Poolside: Laguna-M (Free)": "poolside/laguna-m:free"
}

# Ensure session state variables exist to maintain consistent performance across runs.
if "worker_responses" not in st.session_state:
    st.session_state.worker_responses = {}
if "final_synthesis" not in st.session_state:
    st.session_state.final_synthesis = ""
if "execution_time" not in st.session_state:
    st.session_state.execution_time = 0.0

# -----------------------------------------------------------------------------
# ASYNCHRONOUS API ORCHESTRATION ENGINE
# -----------------------------------------------------------------------------

async def fetch_llm_response(
    session: aiohttp.ClientSession, 
    model_friendly_name: str, 
    model_id: str, 
    api_key: str, 
    messages: list,
    max_retries: int = 5,
    initial_delay: float = 2.0,
    max_delay: float = 16.0,
    backoff_factor: float = 2.0
) -> dict:
    """
    Executes a high-concurrency POST request to the OpenRouter unified API.
    Implements dynamic error isolation and robust exponential backoff with jitter
    to handle rate-limiting (HTTP 429) and transient backend failures (HTTP 5xx).
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-OpenRouter-Title": "Multi-LLM Cross-Reflection Engine",
        "X-OpenRouter-Cache": "true"  # Instructs OpenRouter to leverage edge-level response caching
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.5  # Fixed moderate temperature for stable analytical performance
    }

    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, json=payload, headers=headers, timeout=45) as response:
                status = response.status
                
                if status == 200:
                    data = await response.json()
                    # Validate payload integrity and handle unexpected parsing anomalies
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        return {
                            "friendly_name": model_friendly_name,
                            "model_id": model_id,
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
                            "error": "Malformed JSON payload returned from the API."
                        }
                
                # Check for transient, retryable status codes (Rate limits or server outages)
                elif status in [429, 500, 502, 503, 504]:
                    if attempt == max_retries:
                        return {
                            "friendly_name": model_friendly_name,
                            "model_id": model_id,
                            "success": False,
                            "text": "",
                            "error": f"API Request failed with status {status} after {max_retries} attempts."
                        }
                
                # Treat other statuses (400, 401, 403) as critical and non-retryable
                else:
                    try:
                        err_payload = await response.json()
                        err_msg = err_payload.get("error", {}).get("message", "No message provided")
                    except Exception:
                        err_msg = "Unknown client-side failure."
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
                    "error": f"Request timed out consistently after {max_retries} attempts."
                }
        except Exception as e:
            if attempt == max_retries:
                return {
                    "friendly_name": model_friendly_name,
                    "model_id": model_id,
                    "success": False,
                    "text": "",
                    "error": f"Network transmission error: {str(e)}"
                }
        
        # Calculate next backoff interval applying the compounding factor and uniform jitter
        delay = min(max_delay, initial_delay * (backoff_factor ** (attempt - 1)))
        jitter = random.uniform(0.0, 0.1 * delay)
        await asyncio.sleep(delay + jitter)

    return {
        "friendly_name": model_friendly_name,
        "model_id": model_id,
        "success": False,
        "text": "",
        "error": "Exhausted all backoff retry protocols."
    }

async def execute_parallel_run(selected_workers: dict, api_key: str, prompt: list) -> list:
    """
    Spawns multiple simultaneous asynchronous fetch tasks inside a clean ClientSession,
    reusing TCP connections across all concurrent workers to minimize latency.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_llm_response(session, friendly_name, model_id, api_key, prompt)
            for friendly_name, model_id in selected_workers.items()
        ]
        return await asyncio.gather(*tasks)

# -----------------------------------------------------------------------------
# INTERACTIVE USER INTERFACE CONSTRUCTS
# -----------------------------------------------------------------------------

# Sidebar Panel for API authentication and model inventory mapping
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=70)
    st.title("Engine Config")
    st.markdown("---")
    
    # Secure field for API verification
    openrouter_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        help="Access key generated from your openrouter.ai settings page."
    )
    
    st.markdown("### Model Configuration")
    
    # Interactive multi-select panel for establishing the Worker baseline
    worker_selections = st.multiselect(
        "Choose Phase 1 Workers",
        options=list(AVAILABLE_MODELS.keys()),
        default=[
            "Meta: Llama 3.3 70B Instruct (Free)",
            "Google: Gemini 2.5 Flash (Free)",
            "Tencent: Hy3 MoE 295B (Free)"
        ],
        help="These models will process your input query simultaneously."
    )
    
    # Dropdown selector for establishing the synthesis authority
    judge_selection = st.selectbox(
        "Choose Phase 2 Judge",
        options=list(AVAILABLE_MODELS.keys()),
        index=2,  # Defaults to NVIDIA Nemotron Ultra
        help="This high-capacity model evaluates worker outputs and compiles the consensus."
    )
    
    st.markdown("---")
    st.markdown(
        "<small style='color: grey;'>Orchestrated via Streamlit Community Cloud & OpenRouter Gateway.</small>",
        unsafe_allow_html=True
    )

# Primary Layout Presentation
st.title("🤖 Multi-LLM Cross-Reflection Engine")
st.caption("A production-ready parallel orchestration and peer-review platform for large language models.")

# Primary prompt capture container
user_prompt = st.text_area(
    "Submit your prompt for collaborative cross-reflection:",
    placeholder="e.g., Explain the physical limits of optical photolithography and propose three alternative high-throughput nanostructuring methods.",
    height=120
)

col_ctrl_1, col_ctrl_2 = st.columns([1, 4])
with col_ctrl_1:
    submit_trigger = st.button("Initialize Engine", use_container_width=True, type="primary")

# -----------------------------------------------------------------------------
# TWO-PHASE PIPELINE EXECUTION ORCHESTRATION
# -----------------------------------------------------------------------------

if submit_trigger:
    # Pre-flight credential and selection safety audits
    if not openrouter_key:
        st.error("Engine execution aborted: OpenRouter API key missing. Please insert your credentials in the sidebar.")
    elif not worker_selections:
        st.error("Engine execution aborted: Please select at least one Worker model to initiate Phase 1.")
    elif not user_prompt.strip():
        st.warning("Please input a valid prompt query to kickstart execution.")
    else:
        # Build worker configuration mapping based on UI options
        active_workers = {name: AVAILABLE_MODELS[name] for name in worker_selections}
        judge_model_id = AVAILABLE_MODELS[judge_selection]
        
        # Format the system prompt for processing
        formatted_messages = [{"role": "user", "content": user_prompt}]
        
        # -----------------------------------------------------
        # PHASE 1: SIMULTANEOUS CONCURRENT EXECUTION
        # -----------------------------------------------------
        st.markdown("### Phase 1: Simultaneous Execution")
        progress_tracker = st.empty()
        progress_tracker.info("Establishing connection with OpenRouter. Dispatching requests in parallel...")
        
        start_timestamp = time.time()
        
        # Execute the high-concurrency tasks via nested loop framework
        loop = asyncio.get_event_loop()
        worker_outputs = loop.run_until_complete(
            execute_parallel_run(active_workers, openrouter_key, formatted_messages)
        )
        
        elapsed_workers_time = time.time() - start_timestamp
        progress_tracker.empty()
        
        # Layout worker outputs dynamically in parallel visual columns
        num_columns = len(worker_outputs)
        columns = st.columns(num_columns, border=True)
        
        for idx, col in enumerate(columns):
            response_data = worker_outputs[idx]
            model_header = response_data["friendly_name"]
            
            with col:
                st.markdown(f"##### 🟢 {model_header}")
                if response_data["success"]:
                    st.write(response_data["text"])
                else:
                    st.error(response_data["error"])
        
        # Cache raw worker outcomes in session state for pipeline transfer
        st.session_state.worker_responses = {
            item["friendly_name"]: item["text"] for item in worker_outputs if item["success"]
        }
        
        # -----------------------------------------------------
        # PHASE 2: CROSS-ANALYSIS & COMPILATION BY THE JUDGE
        # -----------------------------------------------------
        if not st.session_state.worker_responses:
            st.error("Phase 2 aborted: All worker tasks failed to deliver output.")
        else:
            st.markdown("---")
            st.markdown("### Phase 2: Cross-Analysis & Synthesis")
            judge_loader = st.empty()
            judge_loader.info(f"Transferring baseline outputs to Judge ({judge_selection}). Analyzing for bias and compiling consensus...")
            
            # Construct the synthetic assessment prompt to instruct the Judge model
            synthesis_system_instruction = (
                "You are an elite research consensus arbitrator. Your primary role is to evaluate answers "
                "from multiple independent expert models. Step-by-step: \n"
                "1. Read the user's original query and all the generated raw answers.\n"
                "2. Conduct a rigorous logical audit. Expose logical flaws, subtle hallucinations, "
                "factual inaccuracies, and structural biases across each response.\n"
                "3. Cross-examine contradictions, highlighting where experts diverge and reconciliate any discrepancies.\n"
                "4. Compile a synthesized master response that integrates the verified strengths of all responses. "
                "Deliver a deeply detailed, structured, clear, and logically flawless master response."
            )
            
            # Compile the baseline content payload for the Judge
            synthesis_input_payload = f"Original Query: {user_prompt}\n\n"
            for source_model, raw_response in st.session_state.worker_responses.items():
                synthesis_input_payload += f"=== Raw Response from [{source_model}] ===\n{raw_response}\n\n"
            
            judge_messages = [
                {"role": "system", "content": synthesis_system_instruction},
                {"role": "user", "content": synthesis_input_payload}
            ]
            
            # Spawn a dedicated client session for the Judge evaluation
            async def run_judge_pipeline():
                async with aiohttp.ClientSession() as session:
                    return await fetch_llm_response(
                        session, judge_selection, judge_model_id, openrouter_key, judge_messages
                    )
            
            judge_response = loop.run_until_complete(run_judge_pipeline())
            
            total_elapsed_time = time.time() - start_timestamp
            st.session_state.execution_time = total_elapsed_time
            judge_loader.empty()
            
            # Display final synthesized results
            if judge_response["success"]:
                st.session_state.final_synthesis = judge_response["text"]
                st.success(f"Synthesis compiled successfully in {st.session_state.execution_time:.2f} seconds.")
                st.markdown("#### Master Consensus & Synthesis Report")
                st.markdown(st.session_state.final_synthesis)
            else:
                st.error(f"Synthesis generation failed: {judge_response['error']}")

# Persistent state presentation logic (Ensures UI elements persist on runtime updates)
elif st.session_state.final_synthesis:
    st.markdown("---")
    st.markdown("### Phase 1: Simultaneous Execution (Cached Data)")
    cached_workers_count = len(st.session_state.worker_responses)
    columns = st.columns(cached_workers_count, border=True)
    
    for idx, (model_header, cached_text) in enumerate(st.session_state.worker_responses.items()):
        with columns[idx]:
            st.markdown(f"##### 🟢 {model_header}")
            st.write(cached_text)
            
    st.markdown("---")
    st.markdown("### Phase 2: Cross-Analysis & Synthesis (Cached Data)")
    st.caption(f"Compiled in {st.session_state.execution_time:.2f} seconds.")
    st.markdown("#### Master Consensus & Synthesis Report")
    st.markdown(st.session_state.final_synthesis)
