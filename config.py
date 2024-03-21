# config.py

# The length of memory stores used for storing prompts and responses
memory_length = 1

# Temperature setting (if used in your model)
temperature = 0.5

# The maximum number of tokens per API call, used in the `selector_prompter` method
max_tokens_per_api_call = 400

# The number of retries to attempt when an API call fails (default value of 3)
max_retries = 3

# CONFIG. The interval for periodic saving in the `start_periodic_save_thread` method (default value of 3600 seconds)
periodic_save_interval = 600  # interval in seconds

# CONFIG. File paths for saving and loading data
faiss_index_path = 'faiss_index.dat'
auxiliary_data_path = 'auxiliary_data.pkl'
meta_data_path = 'meta_data.pkl'

# CONFIG. Number of nearest neighbors to retrieve in operations like searching in a FAISS index
faiss_search_k = 20

# RagHandler model name
rag_model_name = "sentence-transformers/all-MiniLM-L6-v2"
