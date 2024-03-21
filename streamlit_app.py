from venv import logger
import streamlit as st
from amrex import model_prompter
from datetime import datetime
import base64
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


st.set_page_config(
    page_title="AMREx",
    page_icon="🤖",
    layout="wide"
)

st.title("Athena - State Aware AI Assistant")

# Initialize session state for interaction tokens
if 'interaction_tokens' not in st.session_state:
    st.session_state.interaction_tokens = {'sent': 0, 'received': 0}

# Initialize session state for token usage
if 'prev_session_token_usage' not in st.session_state:
    st.session_state.prev_session_token_usage = {'sent': 0, 'received': 0}

# Initialize session state for token count
if 'token_usage' not in st.session_state:
    st.session_state.token_usage = {'sent': 0, 'received': 0}

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to handle sending a message
def send_message(user_message):
    if user_message:
        try:
            model_response = model_prompter.core_processor(user_message)
            logger.debug(f"Streamlit model response: {model_response}")

            # Save the current session token count
            st.session_state.prev_session_token_usage = st.session_state.token_usage.copy()

            # Update the session state with token usage
            session_token_usage = model_prompter.session_token_usage
            st.session_state.token_usage['sent'] += session_token_usage['sent']
            st.session_state.token_usage['received'] += session_token_usage['received']

            # Calculate interaction tokens
            interaction_tokens_sent = st.session_state.token_usage['sent'] - st.session_state.prev_session_token_usage['sent']
            interaction_tokens_received = st.session_state.token_usage['received'] - st.session_state.prev_session_token_usage['received']
            st.session_state.interaction_tokens = {'sent': interaction_tokens_sent, 'received': interaction_tokens_received}

            return model_response

        except Exception as e:
            st.session_state.chat_history += f"Model: An error occurred: {e}\n" # TODO: REVISE THIS

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box 
user_input = st.chat_input("Your Message")
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Send user message to model and display model response start spinner while model is thinking
    with st.spinner("Athena is thinking..."):
        assistant_response = send_message(user_input)
        if assistant_response:
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
            # Display the chat history
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

# Function to format chat history in Markdown
def format_chat_history_to_markdown(chat_history):
    markdown_text = ""
    for message in chat_history:
        role = "You" if message["role"] == "user" else "AMREx"
        markdown_text += f"**{role}:** {message['content']}\n\n"
    return markdown_text

# Function to create a download link with date and time in filename
def create_download_link(markdown_text, base_filename="chat_history"):
    # Getting current date and time
    current_datetime = datetime.now()
    # Formatting date and time (e.g., "2024-01-14_12-30-00")
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # Creating filename with date and time
    filename = f"{base_filename}_{formatted_datetime}.md"
    # Encoding the markdown text
    b64 = base64.b64encode(markdown_text.encode()).decode()
    # Returning the download link with the new filename
    return f'<a href="data:file/markdown;base64,{b64}" download="{filename}">Download Chat History</a>'

with st.sidebar:
    # Safe access for token_usage
    total_sent = st.session_state.token_usage.get('sent', 0)
    total_received = st.session_state.token_usage.get('received', 0)
    with st.sidebar:
        st.write("Total tokens -")
        st.write(f"Sent: {total_sent}, Received: {total_received}")

    # Safe access for interaction_tokens
    interaction_sent = st.session_state.interaction_tokens.get('sent', 0)
    interaction_received = st.session_state.interaction_tokens.get('received', 0)
    with st.sidebar:
        st.write("Tokens in current interaction -")
        st.write(f"Sent: {interaction_sent}, Received: {interaction_received}")

    # Save chat data button in sidebar
    if st.button('Save Chat'):
        try:
            model_prompter.rag_handler.save_faiss_index('faiss_index.dat')
            model_prompter.rag_handler.save_auxiliary_data('auxiliary_data.pkl')
            st.write("Chat data saved successfully on: \n" + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        except Exception as e:
            st.write(f"Error saving chat data: {e}")

    # Load chat export button in sidebar
    if st.button('Export Chat as Markdown'):
        markdown_chat_history = format_chat_history_to_markdown(st.session_state.chat_history)
        st.sidebar.markdown(create_download_link(markdown_chat_history), unsafe_allow_html=True)