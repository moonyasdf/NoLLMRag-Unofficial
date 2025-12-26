import requests
import json
import logging
from typing import List, Optional, Tuple, Set
from .config import Config
from .text_processor import TextProcessor
from .vector_store import VectorStore
from .graph_engine import GraphEngine

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NoLLMRAGPipeline:
    def __init__(self):
        Config.ensure_dirs()
        self.tp = TextProcessor()
        self.vs = VectorStore()
        self.ge = GraphEngine()
        
        if self.ge.load():
            logger.info("Knowledge Graph loaded successfully from disk.")
            if self.vs.load():
                logger.info("Vector indices loaded successfully from disk.")
        else:
            logger.warning("No existing index found. System initialized in empty state.")

    def index(self, documents: List[str]):
        """
        Builds the graph and vector indices for a provided list of documents.
        """
        if not documents:
            logger.error("Indexing failed: Document list is empty.")
            return

        logger.info(f"Starting indexing pipeline for {len(documents)} documents...")
        
        # 1. Build Graph
        tokens_list, chunks_list = self.ge.build(documents, self.tp)
        
        # 2. Create Vector Indices (Tokens and Chunks)
        self.vs.create_indices(tokens_list, chunks_list)
        
        # 3. Persist to disk
        self.ge.save()
        self.vs.save()
        logger.info("Pipeline Indexing Complete. All indices saved to disk.")

    def query(self, query_text: str) -> str:
        """
        Full RAG Pipeline: Retrieval -> Ranking -> Generation.
        """
        if not query_text.strip():
            return "Error: Empty query provided."

        # 1. Map query tokens to Graph nodes
        candidate_tids = self._map_query_to_token_nodes(query_text)
        if not candidate_tids: 
            return "No relevant information found in the knowledge base."

        # 2. Graph-based retrieval (Clustering/Union mode)
        chunk_ids = self.ge.extract_keywords_and_cluster(list(candidate_tids))
        
        # 3. Optimized Ranking (Using pre-calculated vectors from FAISS)
        num_docs = len(self.ge.graph.vs.select(type_eq="document"))
        chunk_ids_with_text = [(cid, self.ge.chunk_data[cid]) for cid in chunk_ids if cid in self.ge.chunk_data]
        
        if not chunk_ids_with_text: 
            return "No relevant text fragments retrieved for this query."
        
        ranked_chunks = self.vs.rank_chunks(query_text, chunk_ids_with_text, num_docs)
        top_chunks_data = ranked_chunks[:Config.TOP_K_CHUNKS]
        
        if not top_chunks_data: 
            return "Information filtered out during semantic ranking."

        # 4. Generate response using LLM
        return self._generate(query_text, top_chunks_data)

    def retrieve_only(self, query_text: str) -> List[str]:
        """
        Retrieval evaluation method. Skips LLM generation for speed and cost.
        """
        candidate_tids = self._map_query_to_token_nodes(query_text)
        if not candidate_tids: 
            return []

        chunk_ids = self.ge.extract_keywords_and_cluster(list(candidate_tids))
        
        num_docs = len(self.ge.graph.vs.select(type_eq="document"))
        chunk_ids_with_text = [(cid, self.ge.chunk_data[cid]) for cid in chunk_ids if cid in self.ge.chunk_data]
        
        if not chunk_ids_with_text: 
            return []
        
        # Optimized Ranking
        ranked = self.vs.rank_chunks(query_text, chunk_ids_with_text, num_docs)
        return [item[0] for item in ranked[:Config.TOP_K_CHUNKS]]

    def retrieve_naive(self, query_text: str, k: int = 30) -> List[str]:
        """
        Baseline Naive RAG: Vector similarity search only, bypassing the graph.
        """
        q_emb = self.vs.model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
        D, I = self.vs.chunk_index.search(q_emb, k)
        
        num_docs = len(self.ge.graph.vs.select(type_eq="document"))
        retrieved_texts = []
        for idx in I[0]:
            if idx != -1:
                chunk_id = int(idx) + num_docs
                if chunk_id in self.ge.chunk_data:
                    retrieved_texts.append(self.ge.chunk_data[chunk_id])
        return retrieved_texts

    def _map_query_to_token_nodes(self, query_text: str) -> Set[int]:
        """
        Maps raw query text to existing token nodes in the graph using vector search.
        """
        q_tokens_raw = self.tp.process_chunk_sequence(query_text)
        if not q_tokens_raw:
            return set()

        candidate_tids = set()
        token_start_id = self.ge.token_start_id
        max_graph_id = self.ge.graph.vcount() - 1
        
        for t_str in q_tokens_raw:
            faiss_indices, _ = self.vs.search_tokens(t_str, top_k=1)
            
            if len(faiss_indices) > 0 and faiss_indices[0] != -1:
                graph_node_id = token_start_id + int(faiss_indices[0])
                
                # Validation: Ensure node exists and is a 'token' node
                if graph_node_id <= max_graph_id:
                    if self.ge.graph.vs[graph_node_id]["type"] == "token":
                        candidate_tids.add(graph_node_id)
                    else:
                        logger.debug(f"Node {graph_node_id} is not a token node. Skipping mapping.")
        
        return candidate_tids

    def retrieve_decomposed(self, sub_questions: List[str], original_query: str) -> List[str]:
        """
        Recupera información procesando cada sub-pregunta por separado y uniendo resultados.
        """
        all_candidate_ids = set()
        
        for sq in sub_questions:
            # 1. Extraer tokens de la sub-pregunta
            q_tokens = self.tp.process_chunk_sequence(sq)
            if not q_tokens: continue
            
            # 2. Mapear a tokens del grafo
            candidate_tids = []
            token_start_id = self.ge.token_start_id
            for t_str in q_tokens:
                faiss_idx, _ = self.vs.search_tokens(t_str, top_k=1)
                if len(faiss_idx) > 0 and faiss_idx[0] != -1:
                    gid = token_start_id + int(faiss_idx[0])
                    if gid < self.ge.graph.vcount() and self.ge.graph.vs[gid]["type"] == "token":
                        candidate_tids.append(gid)
            
            # 3. Obtener chunks via grafo (Modo Unión ya implementado en GraphEngine)
            chunk_ids = self.ge.extract_keywords_and_cluster(candidate_tids)
            all_candidate_ids.update(chunk_ids)
            
        # 4. Ranking Final
        # Usamos la consulta ORIGINAL para ordenar todos los candidatos acumulados
        if not all_candidate_ids: return []
        
        num_docs = len(self.ge.graph.vs.select(type_eq="document"))
        chunk_ids_with_text = [(cid, self.ge.chunk_data[cid]) for cid in all_candidate_ids if cid in self.ge.chunk_data]
        
        ranked = self.vs.rank_chunks(original_query, chunk_ids_with_text, num_docs)
        return [item[0] for item in ranked[:Config.TOP_K_CHUNKS]]

    def _generate(self, query: str, ranked_chunks_data: List[tuple]) -> str:
        """
        Constructs the prompt and calls the Ollama API.
        """
        # Context sorted by descending similarity
        contexts = [item[0] for item in ranked_chunks_data]
        context_block = "\n\n".join(contexts)
        
        prompt = f"""system: You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{context_block}

User: {query}"""
        
        payload = {
            "model": Config.LLM_MODEL, 
            "prompt": prompt, 
            "stream": False
        }
        
        try:
            # Added timeout to prevent hanging in long evaluations
            res = requests.post(Config.OLLAMA_URL, json=payload, timeout=120)
            if res.status_code == 200:
                return res.json().get("response", "Empty response from model.")
            return f"LLM Error {res.status_code}: {res.text}"
            
        except requests.exceptions.Timeout:
            logger.error("LLM Generation request timed out.")
            return "Error: The generation request timed out."
        except Exception as e:
            logger.error(f"Failed to communicate with LLM: {e}")
            return f"Connection Error: {e}"