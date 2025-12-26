import faiss
import numpy as np
import os
import logging
from typing import List, Tuple, Dict
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
        if token_texts:
            token_embs = self.model.encode(token_texts, batch_size=Config.BATCH_SIZE, 
                                           convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
            self.token_index = faiss.IndexFlatIP(token_embs.shape[1])
            self.token_index.add(token_embs)
        
        if chunk_texts:
            chunk_embs = self.model.encode(chunk_texts, batch_size=Config.BATCH_SIZE, 
                                           convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
            self.chunk_index = faiss.IndexFlatIP(chunk_embs.shape[1])
            self.chunk_index.add(chunk_embs)

    def search_tokens(self, query_text: str, top_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        q_emb = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.token_index.search(q_emb, top_k)
        return I[0], D[0] 

    def rank_chunks(self, query: str, chunk_ids_with_text: List[Tuple[int, str]], num_docs: int) -> List[Tuple[str, float]]:
        """
        Versión Optimizada: No re-codifica los chunks. Extrae los vectores de FAISS.
        chunk_ids_with_text: Lista de (graph_id, text)
        num_docs: Offset para mapear graph_id a faiss_id
        """
        if not chunk_ids_with_text: return []
        
        # 1. Obtener solo el vector de la QUERY (único encode necesario)
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # 2. Extraer vectores de los candidatos desde el índice FAISS
        # faiss_id = graph_id - num_docs
        faiss_ids = [cid - num_docs for cid, _ in chunk_ids_with_text]
        
        # Obtenemos los vectores directamente de FAISS (Inmediato)
        try:
            c_embs = np.array([self.chunk_index.reconstruct(int(fid)) for fid in faiss_ids])
        except Exception as e:
            logger.error(f"Error reconstruyendo vectores de FAISS: {e}")
            return []

        # 3. Calcular similitud (Producto punto de vectores normalizados = Similitud Coseno)
        scores = np.dot(c_embs, q_emb.T).flatten()
        
        # 4. Ordenar
        results = []
        for i, score in enumerate(scores):
            results.append((chunk_ids_with_text[i][1], float(score)))
            
        return sorted(results, key=lambda x: x[1], reverse=True)

    def save(self):
        if self.token_index: faiss.write_index(self.token_index, Config.VEC_TOKEN_PATH)
        if self.chunk_index: faiss.write_index(self.chunk_index, Config.VEC_CHUNK_PATH)
        
    def load(self) -> bool:
        if os.path.exists(Config.VEC_TOKEN_PATH):
            self.token_index = faiss.read_index(Config.VEC_TOKEN_PATH)
            self.chunk_index = faiss.read_index(Config.VEC_CHUNK_PATH)
            return True
        return False