import faiss
import numpy as np
import os
import logging
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from .config import Config

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        logger.info(f"Loading Embedding Model: {Config.EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        self.token_index = None
        self.chunk_index = None

    def create_indices(self, token_texts: List[str], chunk_texts: List[str]):
        # 1. Token Index
        if token_texts:
            logger.info(f"Encoding {len(token_texts)} tokens...")
            token_embs = self.model.encode(
                token_texts, 
                batch_size=Config.BATCH_SIZE,
                convert_to_numpy=True, 
                show_progress_bar=True,
                normalize_embeddings=True 
            )
            self.token_index = faiss.IndexFlatIP(token_embs.shape[1])
            self.token_index.add(token_embs)
        
        # 2. Chunk Index
        if chunk_texts:
            logger.info(f"Encoding {len(chunk_texts)} chunks...")
            chunk_embs = self.model.encode(
                chunk_texts, 
                batch_size=Config.BATCH_SIZE,
                convert_to_numpy=True, 
                show_progress_bar=True,
                normalize_embeddings=True
            )
            self.chunk_index = faiss.IndexFlatIP(chunk_embs.shape[1])
            self.chunk_index.add(chunk_embs)

    def search_tokens(self, query_text: str, top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if self.token_index is None:
            raise RuntimeError("Token index not initialized.")
            
        q_emb = self.model.encode(
            [query_text], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        D, I = self.token_index.search(q_emb, top_k)
        return I[0], D[0] 

    def rank_chunks(self, query: str, chunk_texts: List[str]) -> List[Tuple[str, float]]:
        if not chunk_texts: return []
        
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        c_embs = self.model.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True)
        
        scores = np.dot(c_embs, q_emb.T).flatten()
        ranked_indices = np.argsort(scores)[::-1]
        
        return [(chunk_texts[i], float(scores[i])) for i in ranked_indices]

    def save(self):
        if self.token_index:
            faiss.write_index(self.token_index, Config.VEC_TOKEN_PATH)
        if self.chunk_index:
            faiss.write_index(self.chunk_index, Config.VEC_CHUNK_PATH)
        
    def load(self) -> bool:
        if os.path.exists(Config.VEC_TOKEN_PATH):
            self.token_index = faiss.read_index(Config.VEC_TOKEN_PATH)
            self.chunk_index = faiss.read_index(Config.VEC_CHUNK_PATH)
            return True
        return False