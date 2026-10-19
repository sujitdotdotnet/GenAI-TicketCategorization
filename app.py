import json
import os
import re

import streamlit as st
from openai import OpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()  # loads variables from a .env file in the working directory
except ImportError:
    pass  # python-dotenv not installed; real environment variables still work

st.set_page_config(page_title="Support Ticket Categorization System")

# ---------------------------------------------------------------------------
# Provider configuration (all use OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------
PROVIDERS = {
    "OpenRouter": {
        "env_var": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
    },
    "OpenAI": {
        "env_var": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "Claude": {
        "env_var": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/",
        "default_model": "claude-sonnet-5-5",
    },
    "Gemini": {
        "env_var": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
    },
    "DeepSeek": {
        "env_var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
}

# ---------------------------------------------------------------------------
# Sidebar: instructions, provider, API key, model
# ---------------------------------------------------------------------------
st.sidebar.header("How to use")
st.sidebar.markdown(
    """
1. Pick an **LLM provider**.
2. Enter your **API key** for that provider  
   (or leave it empty to use the key from your `.env` file).
3. Paste a support ticket on the main page and click **Process Ticket**.

The Base URL changes automatically with the provider.

**`.env` fallback variables**
- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
"""
)
st.sidebar.divider()
st.sidebar.header("LLM Settings")

provider = st.sidebar.selectbox("LLM provider", list(PROVIDERS.keys()))
cfg = PROVIDERS[provider]

# Base URL updates automatically with the provider (read-only display)
st.sidebar.text_input("Base URL", value=cfg["base_url"], disabled=True, key=f"base_{provider}")

# Separate widget keys per provider so each provider remembers its own key/model
ui_key = st.sidebar.text_input(
    f"API key for {provider}",
    type="password",
    placeholder="Leave empty to use the key from .env",
    key=f"key_{provider}",
)
model_name = st.sidebar.text_input(
    "Model", value=cfg["default_model"], key=f"model_{provider}"
)

# Key priority: value typed in the box -> fallback to .env / environment
env_key = (os.environ.get(cfg["env_var"]) or "").strip()
api_key = ui_key.strip() or env_key

if ui_key.strip():
    st.sidebar.success(f"Using the {provider} key entered above.")
elif env_key:
    st.sidebar.info(f"Using {cfg['env_var']} from .env.")
else:
    st.sidebar.error(
        f"No API key found for {provider}. Add one in the box above "
        f"or set {cfg['env_var']} in .env."
    )

MAX_TOKENS = 500


def get_client():
    """Build an OpenAI-compatible client for the selected provider."""
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def query_llm(prompt, query):
    """Send a system prompt + user query to the selected provider and return text."""
    response = get_client().chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


def parse_json(text):
    """Parse JSON from a model reply, tolerating ```json fences or extra text."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def run_json_task(prompt, query, label):
    """Call the LLM and parse a JSON reply; show errors in the UI and return None on failure."""
    response_text = ""
    try:
        response_text = query_llm(prompt, query)
        return parse_json(response_text)
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from model response ({label}): {e}")
        st.text(f"Raw response: {response_text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during {label}: {e}")
        return None


def classify_ticket(prompt, query):
    return run_json_task(prompt, query, "classification")


def extract_metadata(prompt, query):
    return run_json_task(prompt, query, "metadata extraction")


def predict_priority(prompt, query, problem_type, user_impact):
    full_query = f"""
    Support Ticket: {query}
    Problem Type: {problem_type}
    User Impact: {user_impact}
    Based on the support ticket, problem type, and user impact, predict the priority: Low, Medium, High, or Urgent.
    Return only a structured JSON output in the following format:
    {{"priority": "priority_prediction"}}
    """
    return run_json_task(prompt, full_query, "priority prediction")


def generate_response(response_prompt, query, category, metadata_tags, priority):
    user_message = f"""
    Support Ticket: {query}
    Category: {category}
    Metadata Tags: {metadata_tags}
    Priority: {priority}
    """
    try:
        return query_llm(response_prompt, user_message)
    except Exception as e:
        st.error(f"An unexpected error occurred during response generation: {e}")
        return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
classification_prompt = """
 You are a technical assistant. Classify the support ticket based on the Support Ticket Text presented in the input into the following categories and not any other.
      - Technical issues
      - Hardware issues
      - Data recovery
  Return only a structured JSON output in the following format:
  {"Category": "category_prediction"}
"""

metadata_prompt = """
You are an intelligent assistant that extracts structured metadata from technical support queries.
Analyze the query and extract the following information:

* Device (e.g., Laptop, Phone, Router, etc.)
* Problem Type (e.g., Not Turning On, Lost Internet, Deleted Files)
* User Impact - Estimate based on how severely the issue affects the user's ability to continue working or using the device:

    - * Major: The user cannot proceed with work at all.
    - * Moderate: The user is impacted but may have a workaround.
    - * Minor: The issue is present but does not significantly hinder usage.

Use the following examples as guidance.

Query Text: My phone battery is draining rapidly even on battery saver mode. I barely use it and it drops 50% in a few hours.
Output: {"Device": "Phone", "Problem Type": "Battery Draining", "User Impact": "Minor"}

Query Text: I accidentally deleted a folder containing all project files. Please help me recover it.
Output: {"Device": "Laptop", "Problem Type": "Deleted Files", "User Impact": "Major"}

Query Text: My router is not working.
Output: {"Device": "Router", "Problem Type": "Lost Internet", "User Impact": "Moderate"}

Return the final output only in a valid JSON format without any additional explanation.
"""

priority_prompt = """
You are an intelligent assistant that determines the priority level of a support ticket.
For any given ticket, follow this step-by-step reasoning process to assign the correct priority level: Low, Medium, High.
Step-by-step Evaluation:
Is the device or service completely unusable?
Is the issue blocking critical or time-sensitive work?
Is there a specific deadline or urgency mentioned by the user?
Does the user mention partial functionality or ongoing work?
Is the tone or language expressing frustration or emergency?
After evaluating each step, decide the most appropriate priority level based on the impact and urgency.
Finally, return only the structured output in valid JSON format, like this:
{"priority": "High"}
Do not include your reasoning in the output — just the JSON.
"""

response_prompt = """
You are provided with a support ticket's text along with its Category, Tags, and assigned Priority level.

Follow these steps before generating your final response:

1. Analyze the ticket text to understand the customer's sentiment and main concern.
2. Identify the issue type using the provided Category and Tags.
3. Determine the appropriate ETA based on the Priority level.
4. Compose a short, empathetic response that reassures the customer, acknowledges their concern, and includes the ETA.

Ensure the final response:

1. Is under 50 words
2. Has a polite and empathetic tone
3. Addresses the issue clearly

Return only the final response to the customer. Do not include your reasoning steps in the output.
"""

# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
st.title("Support Ticket Categorization System")
st.caption(f"Using **{provider}** · `{cfg['base_url']}` · model `{model_name}`")

st.write("Enter the support ticket text below:")
support_ticket_input = st.text_area("Support Ticket Text", height=200)

if st.button("Process Ticket"):
    if not api_key:
        st.error(
            f"No API key found for {provider}. Enter one in the sidebar "
            f"or set {cfg['env_var']} in your .env file, then try again."
        )
    elif not support_ticket_input:
        st.warning("Please enter support ticket text to process.")
    else:
        with st.spinner("Processing..."):
            # Categorization
            category_result = classify_ticket(classification_prompt, support_ticket_input)
            category = category_result.get("Category") if category_result else "N/A"

            # Metadata Extraction
            metadata_result = extract_metadata(metadata_prompt, support_ticket_input)
            device = metadata_result.get("Device") if metadata_result else "N/A"
            problem_type = metadata_result.get("Problem Type") if metadata_result else "N/A"
            user_impact = metadata_result.get("User Impact") if metadata_result else "N/A"

            # Priority Prediction
            priority_result = predict_priority(
                priority_prompt, support_ticket_input, problem_type, user_impact
            )
            priority = priority_result.get("priority") if priority_result else "N/A"

            # Draft Response Generation
            draft_response = generate_response(
                response_prompt, support_ticket_input, category, metadata_result, priority
            )

        st.subheader("Category:")
        st.write(category)

        st.subheader("Metadata:")
        st.write(f"Device: {device}")
        st.write(f"Problem Type: {problem_type}")
        st.write(f"User Impact: {user_impact}")

        st.subheader("Priority:")
        st.write(priority)

        st.subheader("Draft Response:")
        st.write(draft_response)