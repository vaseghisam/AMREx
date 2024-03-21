import json
from math import log
import re
import logging
import traceback
import os
import pickle
import numpy as np
import threading
import time
import uuid
from datetime import datetime

import faiss
import openai
from dotenv import load_dotenv
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from config import *

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv('HF_TOKEN')
assistant_id = os.getenv("ASSISTANT_ID_3")

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Set the logging level for 'httpx' to WARNING to suppress INFO messages
# this would avoid logger.info for OpenAI API calls)
logging.getLogger('httpx').setLevel(logging.WARNING)


def handle_api_errors(max_retries):
    """
    A decorator to handle errors from OpenAI API calls.

    This decorator wraps around a function that makes API calls to the OpenAI API.
    It handles specific OpenAI exceptions by retrying the function call a specified
    number of times before giving up.

    Parameters:
    - max_retries (int): The maximum number of retries allowed for the API call.

    Returns:
    - wrapper (function): The wrapped function with error handling.

    Raises:
    - Exception: Raises an exception if the maximum number of retries is reached.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except openai.RateLimitError as e:
                    print(f"OpenAI API request exceeded rate limit: {e}. Retrying after a delay...")
                    retries += 1
                    time.sleep(10)
                except openai.APIConnectionError as e:
                    print(f"Failed to connect to OpenAI API: {e}. Retrying...")
                    retries += 1
                    time.sleep(5)
                except openai.APIError as e:
                    print(f"OpenAI API returned an API Error: {e}. Retrying...")
                    retries += 1
                    time.sleep(5)
            raise Exception("Max retries reached. Unable to get a successful response from the API.")
        return wrapper
    return decorator


class OpenAI_API_Helper:
    """
    A helper class to interact with the OpenAI API.

    This class encapsulates methods for creating threads, sending messages,
    creating and waiting on runs, and getting messages from the OpenAI API.
    It uses a decorator to handle API errors.

    Attributes:
    - assistant_id (str): Identifier for the specific OpenAI assistant.
    - selector: A reference to a selector object (not explicitly defined here).
    - max_retries (int): Maximum number of retries for API calls.
    - client: Reference to the OpenAI API client.
    """

    def __init__(self, assistant_id, selector, max_retries=3):
        self.assistant_id = assistant_id
        self.selector = selector
        self.max_retries = max_retries
        self.client = openai

    @handle_api_errors(max_retries=3)
    def create_thread(self):
        response = self.client.beta.threads.create()
        logger.debug(f"Thread created: {response}")
        return response

    @handle_api_errors(max_retries=3)
    def send_message(self, thread_id, message_content):
        return self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message_content
        )

    @handle_api_errors(max_retries=3)
    def create_run(self, thread_id):
        return self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant_id
        )

    @handle_api_errors(max_retries=3)
    def wait_on_run(self, run, run_id, thread_id, prompt_memory_chain, response_memory_chain, original_prompt):
        timeout = 600
        start_time = time.time()

        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            print(f"Current Run Status: {run.status}, Elapsed Time: {time.time() - start_time}s")

            time.sleep(5)
        return run
    
    @handle_api_errors(max_retries=3)
    def jexon(self, text):
        start_pos = None
        braces_count = 0
        
        for i, char in enumerate(text):
            if char == '{':
                if start_pos is None:
                    start_pos = i
                braces_count += 1
            elif char == '}':
                braces_count -= 1
                if braces_count == 0 and start_pos is not None:
                    potential_json = text[start_pos:i+1]
                    try:
                        json.loads(potential_json)  # Validate JSON
                        yield potential_json  # Yield the valid JSON string
                    except json.JSONDecodeError:
                        pass
                    start_pos = None
        
    @handle_api_errors(max_retries=3)
    def get_messages(self, thread_id):
        messages_object = self.client.beta.threads.messages.list(
            thread_id=thread_id,
            limit=1,
            order='desc'
        )

        assistant_message = {}
        if messages_object.data and len(messages_object.data) > 0:
            latest_message = messages_object.data[0]
            logger.debug(f"Latest message: {latest_message}")

            if latest_message.role == 'assistant':
                if latest_message.content and len(latest_message.content) > 0:
                    content = latest_message.content[0]
                    if hasattr(content, 'text') and hasattr(content.text, 'value'):
                        raw_text = content.text.value
                        logger.debug(f"Raw Text: {raw_text}")

                        json_generator = self.jexon(raw_text)
                        json_string = next(json_generator, None)
                        if json_string:
                            try:
                                assistant_message = json.loads(json_string)
                                # Replace empty strings with None
                                for key in assistant_message:
                                    if assistant_message[key] == "" or assistant_message[key] == "None":
                                        assistant_message[key] = None
                                logger.debug(f"Assistant message: {assistant_message}")
                            except json.JSONDecodeError as e:
                                logger.error(f"Error decoding JSON: {e}")
                        else:
                            logger.debug("No valid JSON found")

        return assistant_message

class RagHandler:
    """
    Class to handle operations related to the RAG (Retrieval-Augmented Generation) model.

    This class provides functionalities to embed text, manage a FAISS index,
    retrieve relevant context, and save or load data associated with the RAG model.

    Attributes:
    - tokenizer: Tokenizer from the Hugging Face library for the RAG model.
    - model: The RAG model from Hugging Face.
    - index: FAISS index for efficient similarity search.
    - index_to_entry (dict): Mapping from FAISS index to data entries.
    - faiss_index_path (str): File path for saving/loading the FAISS index.
    - auxiliary_data_path (str): File path for saving/loading auxiliary data.
    """
    def __init__(self, rag_model_name, faiss_index_path, auxiliary_data_path):

        self.model = SentenceTransformer(rag_model_name)

        # Initialize the base FAISS index with dimensionality of the embeddings
        embedding_dim = 384  # Dimensionality of all-MiniLM-L6-v2 embeddings
        self.base_index = faiss.IndexFlatL2(embedding_dim)

        # Wrap the base index with IndexIDMap to support ID mapping
        self.index = faiss.IndexIDMap(self.base_index)
        print("FAISS index initialized.")

        self.index_to_entry = {}

        self.faiss_index_path = faiss_index_path
        self.auxiliary_data_path = auxiliary_data_path

        self.load_faiss_index(faiss_index_path)
        self.load_auxiliary_data(auxiliary_data_path)
    
    def generate_interaction_id(self, date, time):
        # Create a unique ID by combining date and time
        interaction_id = f"{date}_{time}".replace('-', '').replace(':', '').replace(' ', '')
        # Convert interaction_id to integer (Python int not np.int64)
        interaction_id = int(interaction_id)
        return interaction_id  # Ensure ID is Python int not np.int64

    def embed_text(self, text):
        try:
            # Generate embeddings using Sentence Transformers
            embedded_text = self.model.encode([text], convert_to_numpy=True)
            return embedded_text.astype('float32')
        except Exception as e:
            print(f"Error in text embedding: {e}")
            return None
 
    def add_to_memory(self, entry):
        """Add an entry to the FAISS index after embedding."""
        # Embed the text
        combined_text = f"{entry['Prompt']} {entry['Response']}"
        embedded_text = self.embed_text(combined_text)
        logger.info(f"\nEmbedding vector sample (add_to_memory): {embedded_text[0][:10]}")

        if embedded_text is not None:
            logger.info(f"\nAdding to FAISS index, shape: {embedded_text.shape}, dtype: {embedded_text.dtype}")
        else:
            print("Failed to add to FAISS index: Embedded text is None")
            return

        # Generate a unique interaction ID from date and time
        interaction_id = self.generate_interaction_id(entry['Date'], entry['Time'])
        logger.debug(f"Generated ID for Entry: {interaction_id}")
        # Convert interaction_id directly to Python int (not np.int64)
        faiss_id = int(interaction_id)
        logger.debug(f"Generated FAISS ID (Python int, not np.int64): {faiss_id}")

        # Changed: Using faiss_id as the key in index_to_entry instead of index_position
        if faiss_id in self.index_to_entry:
            logger.warning(f"Overwriting existing entry for ID {faiss_id}: {self.index_to_entry[faiss_id]}")

        logger.debug("About to add to FAISS index.")
        # Add the vector and its ID to the FAISS index
        self.index.add_with_ids(embedded_text.reshape(1, -1), np.array([faiss_id], dtype='int'))
        logger.debug("Entry added to FAISS index.")
        logger.debug(f"New index size: {self.index.ntotal}")

        # Map the FAISS ID to the entry in index_to_entry
        self.index_to_entry[faiss_id] = entry  # Changed: Using faiss_id as key
        logger.debug("Mapped FAISS ID to entry successfully.")
   
    def get_id_range_for_temporal_query(self, start_datetime, end_datetime):
        # Include both date and time in the ID
        # Format the start and end datetime to match the format used in generate_interaction_id
        logger.info(f"\n\nStart Date: {start_datetime}, End Date: {end_datetime}")

        start_id_str = start_datetime.replace('-', '').replace(':', '').replace('T', '').strip()
        end_id_str = end_datetime.replace('-', '').replace(':', '').replace('T', '').strip()

        # Check if time component is included,
        # if not, append default start and end times
        if len(start_id_str) == 8:  # Only date is provided
            start_id_str += "000000"  # Start of the day
        if len(end_id_str) == 8:  # Only date is provided
            end_id_str += "235959"  # End of the day

        # Convert to integers
        start_id = int(start_id_str)
        end_id = int(end_id_str)

        # Range Validity Check
        if start_id >= end_id:
            raise ValueError("Start ID must be less than End ID")
        
        logger.info(f"\n\nID Range for Query: Start ID = {start_id}, End ID = {end_id}")

        return start_id, end_id

    def get_retrieved_context(self, prompt_data, faiss_search_k):
        query = prompt_data.get("query")
        print(f"\n\nQuery: {query}")

        # Extract the full start and end datetime strings
        start_datetime = prompt_data.get("start_datetime", "")
        end_datetime = prompt_data.get("end_datetime", "")
        logger.info(f"\nStart DateTime: {start_datetime}, End DateTime: {end_datetime}")

        # Process datetime strings
        if start_datetime:
            start_datetime = start_datetime.replace('-', '').replace(':', '').replace('T', '').strip()
        if end_datetime:
            end_datetime = end_datetime.replace('-', '').replace(':', '').replace('T', '').strip()

        retrieved_entries = []
        indices = []

        try:
            if query and query != 'None':
                # Embed the query text
                query_embedding = self.embed_text(query)
                logger.debug(f"Embedding vector sample (get_retrieved_context): {query_embedding[0][:10]}")
                _, indices = self.index.search(query_embedding, faiss_search_k)
                indices = indices[0].astype(np.int64)
                logger.debug(f"FAISS search returned indices: {indices}")
            else:
                indices = list(self.index_to_entry.keys())

            if start_datetime and end_datetime:
                # Convert start and end dates to ID range
                start_id, end_id = self.get_id_range_for_temporal_query(start_datetime, end_datetime)
                for idx in indices:
                    if start_id <= idx <= end_id:
                        entry = self.index_to_entry.get(idx)
                        if entry:
                            retrieved_entry = {
                                "prompt": entry.get('Prompt'),
                                "response": entry.get('Response'),
                                "date": entry.get('Date'),
                                "time": entry.get('Time'),
                                "chat_id": entry.get('ChatID')
                            }
                            retrieved_entries.append(retrieved_entry)
                            logger.info(f"Retrieved entry: {retrieved_entry}")
        except Exception as e:
            logger.error(f"Error occurred in get_retrieved_context: {e}")

        logger.debug(f"Total number of entries retrieved: {len(retrieved_entries)}")
        return retrieved_entries


    def save_faiss_index(self, faiss_index_path):
        faiss.write_index(self.index, faiss_index_path)

    def save_auxiliary_data(self, auxiliary_data_path):
        with open(auxiliary_data_path, 'wb') as f:
            pickle.dump(self.index_to_entry, f)

    def load_faiss_index(self, faiss_index_path):
        if os.path.exists(faiss_index_path):
            # Load the index from the file
            self.index = faiss.read_index(faiss_index_path)

            # Check if the loaded index is already an IndexIDMap
            if not isinstance(self.index, faiss.IndexIDMap):
                print("Loaded index is not of type IndexIDMap. Wrapping it now.")
                self.index = faiss.IndexIDMap(self.index)
                print("FAISS index loaded and re-wrapped with IndexIDMap.")
            else:
                print("Loaded index is already an IndexIDMap.")

    def load_auxiliary_data(self, auxiliary_data_path):
        if os.path.exists(auxiliary_data_path):
            with open(auxiliary_data_path, 'rb') as f:
                self.index_to_entry = pickle.load(f)


def handle_memory_errors(func):
    """
    A decorator to handle errors in memory management operations.

    This decorator wraps around a function that performs operations on memory stores.
    It catches any exceptions that occur and logs them, returning None as the output.

    Parameters:
    - func (function): The function to be wrapped.

    Returns:
    - wrapper (function): The wrapped function with error handling.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in memory management operation '{func.__name__}': {e}")
            return None
    return wrapper

class AMRExMemoryManager:
    """
    Class to manage the memory stores for the AMREx model.

    This class maintains prompt and response memory chains, providing functionalities
    to add entries to the memory, print memory stores, and retrieve memory data.

    Attributes:
    - memory_length (int): The maximum length of memory stores.
    - prompt_memory_chain (list): Store for previous prompts.
    - response_memory_chain (list): Store for previous responses.
    """

    def __init__(self, memory_length):
        self.memory_length = memory_length
        self.prompt_memory_chain = []
        self.response_memory_chain = []

    @handle_memory_errors
    def print_memory_stores(self):
        # Displays the current contents of memory stores.
        print("Prompt Memory Store:")
        for i, prompt in enumerate(self.prompt_memory_chain, 1):
            print(f"{i}: {prompt}")

        print("\nResponse Memory Store:")
        for i, response in enumerate(self.response_memory_chain, 1):
            print(f"{i}: {response}")

    @handle_memory_errors
    def add_to_prompt_memory(self, prompt):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.prompt_memory_chain.append((timestamp, prompt))
        if len(self.prompt_memory_chain) > self.memory_length:
            self.prompt_memory_chain.pop(0)

    @handle_memory_errors
    def add_to_response_memory(self, response):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.response_memory_chain.append((timestamp, response))
        if len(self.response_memory_chain) > self.memory_length:
            self.response_memory_chain.pop(0)

    @handle_memory_errors
    def get_memory(self):
        # Return the memory stores
        return self.prompt_memory_chain, self.response_memory_chain



class AMRExMemorySelectorPrompter:
    """
    Class to select relevant memory for the AMREx model.

    This class is responsible for preparing the context, handling thread operations,
    and prompting the model with the appropriate memory depth. It also provides methods
    to check response validity and manage internal responses.

    Attributes:
    - api_helper: An instance of OpenAI_API_Helper for API interactions.
    - rag_handler: An instance of RagHandler for RAG operations.
    - get_chat_id_func (function): Function to retrieve the current chat ID.
    - max_tokens_per_api_call (int): Maximum tokens per API call.
    - faiss_search_k (int): Number of nearest neighbors to search in the FAISS index.
    """

    def __init__(self, api_helper, max_tokens_per_api_call, faiss_search_k, rag_handler=None, get_chat_id_func=None):
        self.api_helper = api_helper
        self.rag_handler = rag_handler
        self.get_chat_id_func = get_chat_id_func
        self.max_tokens_per_api_call = max_tokens_per_api_call
        self.faiss_search_k = faiss_search_k
        self.current_memory_depth = 0 # Initialize current_memory_depth which communicates the current memory depth to the frontend 
        
        self.tokenizer = AutoTokenizer.from_pretrained(rag_model_name)
        self.interaction_token_usage = {'sent': 0, 'received': 0}

    def calculate_token_usage(self, text):
        # Implement the actual token counting logic using self.tokenizer
        return len(self.tokenizer.encode(text))

    def get_and_reset_interaction_tokens(self):
        tokens = self.interaction_token_usage.copy()
        self.interaction_token_usage = {'sent': 0, 'received': 0}
        return tokens

    def handle_thread_operations(self, thread_id, data_to_send,prompt_memory_chain, response_memory_chain, original_prompt):     
        """
        Handles the operations related to managing threads with the OpenAI API.

        This method creates a new thread if not already present, sends messages, creates runs,
        and waits for runs to complete. It then retrieves messages from the completed run.

        Parameters:
        - thread_id (str): The identifier for the thread. If None, a new thread is created.
        - data_to_send (dict): The data to send in the message.
        - prompt_memory_chain (list): The chain of previous prompts.
        - response_memory_chain (list): The chain of previous responses.
        - original_prompt (str): The original user prompt.

        Returns:
        - assistant_message (dict): The message from the assistant.
        - completed_run: The completed run object.
        """ 
        try:
            if thread_id is None:
                thread = self.api_helper.create_thread()
                thread_id = thread.id
                print(f"Thread created. ID: {thread_id}")

            self.api_helper.send_message(thread_id, json.dumps(data_to_send))
            run = self.api_helper.create_run(thread_id)
            completed_run = self.api_helper.wait_on_run(
                run, run.id,
                thread_id,
                prompt_memory_chain,
                response_memory_chain,
                original_prompt
            )

            return self.api_helper.get_messages(thread_id), completed_run
        except Exception as e:
            print(f"Error in thread operations: {e}")
            return None, None

    def prepare_context(self, memory_depth, prompt_memory_chain, response_memory_chain, original_prompt):
        """
        Prepares the context for the model based on the given memory depth.

        This method generates a combined memory string based on the specified memory depth.
        It also retrieves the chat ID, current date, and time for context.

        Parameters:
        - memory_depth (int): The depth of memory to use in context preparation.
        - prompt_memory_chain (list): The chain of previous prompts.
        - response_memory_chain (list): The chain of previous responses.
        - original_prompt (str): The original user prompt.

        Returns:
        - combined_memory (str): The combined memory string.
        - chat_id (str): The current chat ID.
        - current_date (str): The current date.
        - current_time (str): The current time.
        """
        logger.debug("Debug: Prompt Memory Chain: %s", prompt_memory_chain)
        logger.debug("Debug: Response Memory Chain: %s", response_memory_chain)
        try:
            max_memory_depth = 2
            if memory_depth > max_memory_depth:
                return "depth_limit_exceeded", None, None, None

            chat_id = self.get_chat_id_func()
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")

            if memory_depth == 0:
                combined_memory = []
                print(f"\n>>>>>>>COMBINED MEMORY AT 0: {combined_memory}")
            elif memory_depth == 1:
                combined_memory = []
                for (prompt_timestamp, prompt_text), (response_timestamp, response_text) in zip(prompt_memory_chain, response_memory_chain):
                    entry = {
                        'prompt': prompt_text,
                        'response': response_text,
                        'prompt_timestamp': prompt_timestamp,
                        'response_timestamp': response_timestamp,
                        'chat_id': chat_id
                    }
                    combined_memory.append(entry)
                print(f"\n>>>>>>>COMBINED MEMORY AT 1: {combined_memory}")
            elif memory_depth == 2 and self.rag_handler is not None:
                # Additional check to ensure get_retrieved_context is a callable method
                if callable(getattr(self.rag_handler, 'get_retrieved_context', None)):
                    # Getting the RAG content without any modification
                    combined_memory = self.rag_handler.get_retrieved_context(original_prompt, faiss_search_k)
                    print(f"\n>>>>>>>COMBINED MEMORY AT 2: {combined_memory}")
                else:
                    print("Error: get_retrieved_context method is not available in rag_handler.")
                    combined_memory = ""
            else:
                combined_memory = ""

            return combined_memory, chat_id, current_date, current_time

        except Exception as e:
            print(f"Error in preparing context: {e}")   
            traceback.print_exc()  # This will print the full traceback
            return [], "", "", ""  # Default values indicating failure in preparing context
   

    def clear_redundant_responses(self):
        """
        Clears the response and internal response.

        This method is used to reset the response and internal response in the event of an error
        or when transitioning to a different memory layer.

        Returns:
        - Tuple of empty strings.
        """
        return "", ""
    
    def is_valid_internal_response(self, response):
        """
        Checks if the internal response is valid.

        A valid internal response is defined as one that is not None, not empty, and does not contain
        internal error messages.

        Parameters:
        - response (str): The response to check for validity.

        Returns:
        - (bool): True if the response is valid, False otherwise.
        """
        # Check if the response is not None, not empty, and does not contain internal error messages
        if response == "None" or not response:
            return False
        if response and not response.isspace() and "internal error" not in response.lower():
            return True
        return False

    def selector_prompter(self, prompt_memory_chain, response_memory_chain, original_prompt, memory_depth=0, get_chat_id_func=None, arguments=None, thread_id=None):
        """
        Selects relevant memory and prompts the model for a response.

        This method orchestrates the selection of memory based on the specified depth, prepares
        context, handles thread operations, and manages responses.

        Parameters:
        - prompt_memory_chain (list): The chain of previous prompts.
        - response_memory_chain (list): The chain of previous responses.
        - original_prompt (str): The original user prompt.
        - memory_depth (int): The current depth of memory.
        - get_chat_id_func (function): Function to retrieve the chat ID.
        - arguments (dict): Additional arguments, if any.
        - thread_id (str): The ID of the current thread.

        Returns:
        - response (str): The generated response.
        - memory_message (str): A message about the memory status.
        """
        logger.debug("Entering selector_prompter...")
        
        # Update the current_memory_depth which communicates the current memory depth to the frontend 
        self.current_memory_depth = memory_depth

        context_info = self.prepare_context(memory_depth, prompt_memory_chain, response_memory_chain, original_prompt)
        if context_info[0] == "depth_limit_exceeded":
            return None, "Memory depth limit exceeded"

        combined_memory, chat_id, current_date, current_time = context_info
        logger.debug(f"memory_length: {memory_length}")
        data_to_send = {
            "memory_depth": memory_depth,
            "current_prompt": original_prompt,
            "combined_memory": combined_memory,
            "max_tokens_per_api_call": max_tokens_per_api_call,
            "chat_id": chat_id,
            "current_date": current_date,
            "current_time": current_time,
            "memory_length": memory_length
        }

        # Get tokens before sending data
        sent_tokens = self.calculate_token_usage(json.dumps(data_to_send))
        self.interaction_token_usage['sent'] += sent_tokens
        logger.info(f"Tokens sent in this interaction: {sent_tokens}, Total sent: {self.interaction_token_usage['sent']}")

        assistant_message, completed_run = self.handle_thread_operations(
            thread_id,
            data_to_send,
            prompt_memory_chain,
            response_memory_chain,
            original_prompt
        )

        # Get tokens after receiving data
        received_tokens = self.calculate_token_usage(json.dumps(assistant_message))
        self.interaction_token_usage['received'] += received_tokens
        logger.info(f"Tokens received in this interaction: {received_tokens}, Total received: {self.interaction_token_usage['received']}")

        logger.debug(f"\n\nassistant_message: {assistant_message}")
        # Extract the necessary information from assistant_message
        insufficient_context = assistant_message.get('insufficient_context')
        current_prompt = assistant_message.get('current_prompt')
        logger.info(f"\n\ncurrent_prompt: {current_prompt}")
        response = assistant_message.get('response')
        logger.info(f"\n\nresponse: {response}")
        internal_response = assistant_message.get('internal_response')

        memory_message = "Sufficient context retrieved." 
        interim_response = internal_response # Save the interim response before clearing in recursive calls  

        if insufficient_context is not None and 'INSUFFICIENT_CONTEXT' in insufficient_context:

            response, internal_response = self.clear_redundant_responses() # Clear the response and internal_response

            if memory_depth < 1:
                logger.info("")
                logger.info(f"\nInsufficient context in memory layer {memory_depth}, switching to memory layer {memory_depth + 1}.")
                # Handle recursive call potentially returning None
                logger.debug("Before recursive call with memory_depth:", memory_depth)
                recursive_result = self.selector_prompter(
                    prompt_memory_chain,
                    response_memory_chain,
                    original_prompt,
                    memory_depth=memory_depth + 1,
                    thread_id=thread_id
                )
                return recursive_result # Recursive_response, recursive_memory_message
            elif memory_depth == 1:
                logger.info("")
                logger.info(f"\nInsufficient context in memory layer {memory_depth}, switching to memory layer {memory_depth + 1}.")
                # Handle recursive call potentially returning None
                logger.debug("Before recursive call with memory_depth:", memory_depth)
                recursive_result = self.selector_prompter(
                    prompt_memory_chain,
                    response_memory_chain,
                    current_prompt,
                    memory_depth=memory_depth + 1,
                    thread_id=thread_id
                )
                return recursive_result # Recursive_response, recursive_memory_message
            else:
                print("\nInsufficient context in memory layer 0 to 2.")
                memory_message = "Insufficient context in memory layer 0 to 2."
                response = ""
                # Use the validation method to determine the appropriate response
                if not self.is_valid_internal_response(interim_response):
                    DEFAULT_RESPONSE = "Hmm... Something happened or I can't remember. Could you provide more details or ask a different question?"
                    interim_response = DEFAULT_RESPONSE
                return interim_response, memory_message # Interim response is returned here

        # Exception error handling for the case when the response is None and insufficient_context is None
        # if (response is None or response.strip() == "" or response == "None") and (insufficient_context is None or insufficient_context == 'None'):
        if response is None and insufficient_context is None:
            fallback_response = "Hmm... Something happened or I can't remember. Could you provide more details, try again or ask a different question?"
            logger.warning(f"Model returned 'None' or empty response with sufficient context. Prompt: '{original_prompt}'. Falling back to response: '{fallback_response}'")
            
            # Check if internal_response is valid before using it as a fallback
            if self.is_valid_internal_response(internal_response):
                response = internal_response
            else:
                response = fallback_response
        logger.info(f"Returning from selector_prompter.\nresponse: {response}\nmemory message: {memory_message}")
        logger.info(f"Returning from selector_prompter with total interaction tokens: {self.interaction_token_usage}")
        return response, memory_message



class AMRExMain:
    """
    Class to prompt the AMREx model with memory.

    This class integrates memory management and RAG handling functionalities
    to process user inputs and generate responses from the AMREx model. It also
    tracks token usage and manages periodic saving of model data.

    Attributes:
    - memory_manager (AMRExMemoryManager): Manages memory stores.
    - rag_handler (RagHandler): Handles RAG model operations.
    - chat_id (str): Identifier for the chat session.
    - periodic_save_interval (int): Interval for periodic data saving.
    - token_usage (dict): Tracks token usage for sent and received text.
    - tokenizer: Tokenizer from the Hugging Face library for the RAG model.

    Methods:
    - calculate_token_usage: Calculates number of tokens used in a given text.
    - start_periodic_save_thread: Starts background thread for periodic data saving.
    - periodic_save: Saves FAISS index and auxiliary data at regular intervals.
    - start_chat_session: Initializes a new chat session with a unique ChatID.
    - get_chat_id: Retrieves the current chat session ID.
    - update_memory_stores: Updates the prompt and response memory stores.
    - core_processor: Core method to process user prompts and generate responses.
    """

    def __init__(self, memory_manager, rag_handler, assistant_id, periodic_save_interval, max_tokens_per_api_call, faiss_search_k, max_retries):
        self.memory_manager = memory_manager
        self.rag_handler = rag_handler
        self.chat_id = None
        self.periodic_save_interval = periodic_save_interval
        self.session_token_usage = {'sent': 0, 'received': 0}

        # Create the selector instance
        self.selector = AMRExMemorySelectorPrompter(
            api_helper=None,
            rag_handler=self.rag_handler,
            get_chat_id_func=self.get_chat_id,
            max_tokens_per_api_call=max_tokens_per_api_call,
            faiss_search_k=faiss_search_k
        )

        # Create the API helper instance and pass the selector to it
        self.api_helper = OpenAI_API_Helper(
            assistant_id=assistant_id,
            selector=self.selector,
            max_retries=max_retries
        )

        # Update the selector's api_helper attribute to the newly created api_helper instance
        self.selector.api_helper = self.api_helper

        # Start the background thread for periodic saving for RAGHandler
        self.start_periodic_save_thread()  

    def get_interaction_token_usage(self):
        # Fetch the interaction token usage from the selector
        return self.selector.get_and_reset_interaction_tokens()

    def get_current_memory_depth(self):
        """
        Retrieves the current memory depth from the selector,
        while updating the current_memory_depth attribute.    
        current_memory_depth communicates the current memory depth to frontend

        Returns:
            int: The current memory depth.
        """
        # Return the current memory depth from the selector
        return self.selector.current_memory_depth

    def start_periodic_save_thread(self):
        """
        Initializes and starts a background thread for periodic saving of the RAG handler data.

        The thread runs a periodic_save method at intervals specified by periodic_save_interval.
        It is set as a daemon thread so it terminates when the main program exits.
        """
        background_thread = threading.Thread(target=self.periodic_save, args=(self.periodic_save_interval,))
        background_thread.daemon = True # Daemon threads are shut down when main program exits
        background_thread.start()

    def periodic_save(self, periodic_save_interval):
        """
        Periodically saves the FAISS index and auxiliary data.

        This method is meant to be run in a background thread and executes a save operation
        at the specified interval.

        Parameters:
        - periodic_save_interval (int): Time interval (in seconds) for the save operation.
        """
        while True:
            time.sleep(periodic_save_interval)
            self.rag_handler.save_faiss_index('faiss_index.dat')
            self.rag_handler.save_auxiliary_data('auxiliary_data.pkl')
            logger.debug("Auto-saved FAISS data at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def start_chat_session(self):
        """
        Starts a new chat session and generates a unique ChatID.

        This method is used to initiate a new chat session. It sets a unique identifier
        for the chat session based on the current date and time.
        """
        # Generate a new ChatID for this chat session only if it's not already started
        if not self.chat_id:
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.chat_id = str(uuid.uuid4()) + "_" + current_datetime
            print(f"Chat Session Started with ChatID: {self.chat_id}")
   
    def get_chat_id(self):
        """
        Retrieves the ChatID of the current chat session.

        Returns:
        - chat_id (str): The unique identifier of the current chat session.
        """
        return self.chat_id

    def update_memory_stores(self, original_prompt, response):
        """
        Updates the memory stores with the given prompt and response.

        This method adds the latest interaction (prompt and response) to the memory chains.
        It also creates an entry for the RAG handler memory.

        Parameters:
        - original_prompt (str): The user's original prompt.
        - response (str): The generated response to the prompt.
        """
        logger.debug(f"Adding to memory - Prompt: {original_prompt}, Response: {response}")

        # Specific response to be excluded from memory
        exclude_response = "Hmm... Something happened or I can't remember. Could you provide more details or ask a different question?"

        # Replace the response with an empty string if it matches the exclude_response
        memory_response = "" if response == exclude_response else response

        self.memory_manager.add_to_prompt_memory(original_prompt)
        self.memory_manager.add_to_response_memory(memory_response)

        # Log the update
        logger.debug(f"Updated prompt memory chain: {self.memory_manager.prompt_memory_chain}")
        logger.debug(f"Updated response memory chain: {self.memory_manager.response_memory_chain}")

        # Prepare the entry for RAG handler memory, replacing the response if necessary
        entry = {
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Time": datetime.now().strftime("%H:%M:%S"),
            "ChatID": self.get_chat_id(),
            "Prompt": original_prompt,
            "Response": memory_response  # Use modified response
        }
        self.rag_handler.add_to_memory(entry)

    def core_processor(self, original_prompt):
        """
        Processes the given prompt and generates a response.

        This method integrates various components of the AMREx system to process a user's
        prompt. It manages the chat session, token usage, memory updates, and model prompting
        to generate and return a response.

        Parameters:
        - original_prompt (str): The user's original prompt.

        Returns:
        - response (str): The generated response to the prompt.
        """
        # Send the prompt to the model and return the response
        
        # Check if ChatID is set, if not, start a new chat session
        self.start_chat_session()

        # Log the switch to memory layer 0 at the start of a new interaction
        print("\n-------NEW INTERACTION-------\nNow, reset to memory layer 0.")

        # Retrieve prompt and response memory chains
        prompt_memory_chain, response_memory_chain = self.memory_manager.get_memory()
        
        # Select relevant memory and get response from the model
        response, memory_message = self.selector.selector_prompter(
            prompt_memory_chain, response_memory_chain, original_prompt, memory_depth=0
        )
        logger.info(f"Response from selector_prompter: {response}")

        if 'Error' in memory_message:
            return response  # Early return in case of error
        
        # Retrieve and reset interaction tokens
        interaction_tokens = self.selector.get_and_reset_interaction_tokens()
        logger.info(f"Interaction tokens for this call: Sent: {interaction_tokens['sent']}, Received: {interaction_tokens['received']}")

        # Accumulate session token usage
        self.session_token_usage['sent'] += interaction_tokens['sent']
        self.session_token_usage['received'] += interaction_tokens['received']
        logger.info(f"Updated session token usage: Sent: {self.session_token_usage['sent']}, Received: {self.session_token_usage['received']}")

        self.update_memory_stores(original_prompt, response)

        logger.debug("Memory stores updated.")
        # Debug statements to confirm memory chains are updated
        logger.debug("Updated prompt memory: " + str(self.memory_manager.prompt_memory_chain))
        logger.debug("Updated response memory: " + str(self.memory_manager.response_memory_chain))

        # Debug print to check the response format
        logger.debug(f"Response to Streamlit: {response}")

        # Return the response and token usage for this interaction
        return response

# Instantiate AMRExMemoryManager
memory_manager = AMRExMemoryManager(memory_length)

# Instantiate RagHandler
rag_handler = RagHandler(rag_model_name, faiss_index_path, auxiliary_data_path)

# Initialize the model prompter
model_prompter = AMRExMain(
    memory_manager, 
    rag_handler, 
    assistant_id, 
    periodic_save_interval,
    max_tokens_per_api_call,
    faiss_search_k,
    max_retries
)

# Example usage
# response = model_prompter.core_processor("What is the capital of France?")
# print(response)
# Alternatively use the ´ChatInterface´ class to interact with the model

