
import streamlit as st
import pandas as pd
import json
import os
from openai import OpenAI

# Load OpenAI API key and base URL from Colab secrets
try:
    OPENAI_API_KEY = os.environ.get("API_KEY")
    OPENAI_API_BASE = os.environ.get("API_BASE")
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
except Exception as e:
    st.error(f"Error loading OpenAI credentials: {e}")
    st.stop()


# Define the functions for categorization, metadata extraction, priority prediction, and response generation
def query_openai(prompt, query):
    """
    Queries the OpenAI model with a given prompt and query.
    Args:
        prompt (str): The prompt for the model.
        query (str): The query to be answered by the model.
    Returns:
        str: The model's response.
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",  # Or another suitable OpenAI model
        messages=messages,
        max_tokens=100  # Adjust max_tokens as needed
    )
    return response.choices[0].message.content

def classify_ticket(prompt, query):
    """
    Classifies a support ticket using the OpenAI model and returns the result in JSON format.
    Args:
        prompt (str): The classification prompt for the model.
        query (str): The support ticket text to be classified.
    Returns:
        dict: A dictionary containing the classification result, or None if classification fails.
    """
    try:
        response_text = query_openai(prompt, query)
        # Attempt to parse the response text as JSON
        classification_result = json.loads(response_text)
        return classification_result
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from OpenAI response: {e}")
        st.text(f"Raw OpenAI response: {response_text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during classification: {e}")
        return None

def extract_metadata(prompt, query):
    """
    Extracts metadata from a support ticket using the OpenAI model and returns the result in JSON format.
    Args:
        prompt (str): The metadata extraction prompt for the model.
        query (str): The support ticket text to extract metadata from.
    Returns:
        dict: A dictionary containing the extracted metadata, or None if extraction fails.
    """
    try:
        response_text = query_openai(prompt, query)
        # Attempt to parse the response text as JSON
        metadata_result = json.loads(response_text)
        return metadata_result
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from OpenAI response: {e}")
        st.text(f"Raw OpenAI response: {response_text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during metadata extraction: {e}")
        return None

def predict_priority(prompt, query, problem_type, user_impact):
    """
    Predicts the priority of a support ticket using the OpenAI model and returns the result in JSON format.
    Args:
        prompt (str): The priority prediction prompt for the model.
        query (str): The support ticket text to predict the priority for.
        problem_type (str): The extracted problem type.
        user_impact (str): The extracted user impact.
    Returns:
        dict: A dictionary containing the predicted priority, or None if prediction fails.
    """
    try:
        # Include problem_type and user_impact in the query sent to the model
        full_query = f"""
        Support Ticket: {query}
        Problem Type: {problem_type}
        User Impact: {user_impact}
        Based on the support ticket, problem type, and user impact, predict the priority: Low, Medium, High, or Urgent.
        Return only a structured JSON output in the following format:
        {{"priority": "priority_prediction"}}
        """
        response_text = query_openai(prompt, full_query)
        priority_result = json.loads(response_text)
        return priority_result
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from OpenAI response: {e}")
        st.text(f"Raw OpenAI response: {response_text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during priority prediction: {e}")
        return None

def generate_response(response_prompt, query, category, metadata_tags, priority):
    """
    Generates a draft response for a support ticket using the OpenAI model.
    Args:
        response_prompt (str): The prompt for generating the response.
        query (str): The original support ticket text.
        category (str): The predicted category of the ticket.
        metadata_tags (dict): The extracted metadata tags (Device, Problem Type, User Impact).
        priority (str): The predicted priority of the ticket.
    Returns:
        str: The generated response text, or None if response generation fails.
    """
    # Combine the inputs into a single message for the model
    user_message = f"""
    Support Ticket: {query}
    Category: {category}
    Metadata Tags: {metadata_tags}
    Priority: {priority}
    """

    try:
        # Pass the combined message to the query_openai function
        response_text = query_openai(response_prompt, user_message)
        return response_text
    except Exception as e:
        st.error(f"An unexpected error occurred during response generation: {e}")
        return None

# Define the prompts
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

priority_prompt ="""
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


# Streamlit App
st.title("Support Ticket Categorization System")

st.write("Enter the support ticket text below:")

support_ticket_input = st.text_area("Support Ticket Text", height=200)

if st.button("Process Ticket"):
    if support_ticket_input:
        st.write("Processing...")

        # Categorization
        category_result = classify_ticket(classification_prompt, support_ticket_input)
        category = category_result.get('Category') if category_result else "N/A"
        st.subheader("Category:")
        st.write(category)

        # Metadata Extraction
        metadata_result = extract_metadata(metadata_prompt, support_ticket_input)
        device = metadata_result.get('Device') if metadata_result else "N/A"
        problem_type = metadata_result.get('Problem Type') if metadata_result else "N/A"
        user_impact = metadata_result.get('User Impact') if metadata_result else "N/A"

        st.subheader("Metadata:")
        st.write(f"Device: {device}")
        st.write(f"Problem Type: {problem_type}")
        st.write(f"User Impact: {user_impact}")

        # Priority Prediction
        priority_result = predict_priority(priority_prompt, support_ticket_input, problem_type, user_impact)
        priority = priority_result.get('priority') if priority_result else "N/A"
        st.subheader("Priority:")
        st.write(priority)

        # Draft Response Generation
        draft_response = generate_response(response_prompt, support_ticket_input, category, metadata_result, priority)
        st.subheader("Draft Response:")
        st.write(draft_response)


    else:
        st.warning("Please enter support ticket text to process.")
