import os
import multiprocessing

class Config:
    # --- Paper Hyperparameters ---
    CHUNK_SIZE = 100         
    KEYWORD_TAU = 0.5        
    TOP_K_CHUNKS = 30        
    
    # --- System Settings ---
    # Batch size for both SpaCy and Embeddings (CPU friendly default)
    BATCH_SIZE = 200 
    
    # --- Models ---
    SPACY_MODEL = "en_core_web_lg"
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3" 
    
    # --- Hardware Acceleration ---
    try:
        _total_cores = multiprocessing.cpu_count()
        NUM_PROCESSES = max(1, _total_cores - 1)
    except NotImplementedError:
        NUM_PROCESSES = 1
        
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "indices")
    
    GRAPH_PATH = os.path.join(DATA_DIR, "graph.pkl")
    VEC_TOKEN_PATH = os.path.join(DATA_DIR, "token_index.faiss")
    VEC_CHUNK_PATH = os.path.join(DATA_DIR, "chunk_index.faiss")
    METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

    # --- Local LLM ---
    OLLAMA_URL = "http://localhost:11434/api/generate"
    LLM_MODEL = "qwen2.5:7b" 

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.DATA_DIR, exist_ok=True)