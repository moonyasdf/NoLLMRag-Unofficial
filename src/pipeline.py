import requests
import json
import logging
from typing import List, Optional, Tuple
from .config import Config
from .text_processor import TextProcessor
from .vector_store import VectorStore
from .graph_engine import GraphEngine

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NoLLMRAGPipeline:
    def __init__(self):
        Config.ensure_dirs()
        self.tp = TextProcessor()
        self.vs = VectorStore()
        self.ge = GraphEngine()
        
        if self.ge.load():
            logger.info("Graph loaded successfully.")
            if self.vs.load():
                logger.info("Vector indices loaded successfully.")
        else:
            logger.warning("No existing index found. System ready to index new documents.")

    def index(self, documents: List[str]):
        if not documents:
            logger.error("Attempted to index empty document list.")
            return

        logger.info(f"Starting pipeline indexing for {len(documents)} documents.")
        tokens_list, chunks_list = self.ge.build(documents, self.tp)
        self.vs.create_indices(tokens_list, chunks_list)
        self.ge.save()
        self.vs.save()
        logger.info("Pipeline Indexing Complete & Saved.")

    def query(self, query_text: str) -> str:
        if not query_text.strip():
            return "Empty query provided."

        q_tokens_raw = self.tp.process_chunk_sequence(query_text)
        if not q_tokens_raw: 
            return "Query keywords extraction failed (empty semantic tokens)."
            
        candidate_tids = set()
        token_start_id = self.ge.token_start_id
        max_graph_id = self.ge.graph.vcount() - 1
        
        logger.info(f"Mapping {len(q_tokens_raw)} query tokens to graph...")
        for t_str in q_tokens_raw:
            faiss_idx, _ = self.vs.search_tokens(t_str, top_k=1)
            
            if len(faiss_idx) == 0 or faiss_idx[0] == -1:
                continue
                
            graph_node_id = token_start_id + int(faiss_idx[0])
            
            # Defensive check
            if graph_node_id <= max_graph_id:
                if self.ge.graph.vs[graph_node_id]["type"] == "token":
                    candidate_tids.add(graph_node_id)
                else:
                    logger.debug(f"Mapped ID {graph_node_id} is not a token node. Skipping.")
        
        if not candidate_tids: 
            return "No relevant tokens found in knowledge graph."

        chunk_ids = self.ge.extract_keywords_and_cluster(list(candidate_tids))
        
        candidate_chunks = [self.ge.chunk_data[cid] for cid in chunk_ids if cid in self.ge.chunk_data]
        if not candidate_chunks: 
            return "No chunks retrieved from graph clustering."
        
        ranked_chunks = self.vs.rank_chunks(query_text, candidate_chunks)
        top_chunks_data = ranked_chunks[:Config.TOP_K_CHUNKS]
        
        if not top_chunks_data: 
            return "No chunks remained after semantic ranking."

        return self._generate(query_text, top_chunks_data)

    def retrieve_only(self, query_text: str) -> List[str]:
        """Debug method to get raw chunks without LLM generation."""
        q_tokens_raw = self.tp.process_chunk_sequence(query_text)
        candidate_tids = set()
        token_start_id = self.ge.token_start_id
        
        for t_str in q_tokens_raw:
            faiss_idx, _ = self.vs.search_tokens(t_str, top_k=1)
            if len(faiss_idx) > 0 and faiss_idx[0] != -1:
                gid = token_start_id + int(faiss_idx[0])
                # Safety check without spamming logs in debug mode
                if gid < self.ge.graph.vcount() and self.ge.graph.vs[gid]["type"] == "token":
                    candidate_tids.add(gid)
        
        if not candidate_tids: return []
        chunk_ids = self.ge.extract_keywords_and_cluster(list(candidate_tids))
        candidate_chunks = [self.ge.chunk_data[cid] for cid in chunk_ids if cid in self.ge.chunk_data]
        ranked = self.vs.rank_chunks(query_text, candidate_chunks)
        return [item[0] for item in ranked[:Config.TOP_K_CHUNKS]]

    def _generate(self, query: str, ranked_chunks_data: List[tuple]) -> str:
        contexts = [item[0] for item in ranked_chunks_data]
        context_block = "\n\n".join(contexts)
        
        prompt = f"""system: You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{context_block}

User: {query}"""
        
        payload = {"model": Config.LLM_MODEL, "prompt": prompt, "stream": False}
        
        try:
            # FIX: Added timeout to prevent infinite hanging
            res = requests.post(Config.OLLAMA_URL, json=payload, timeout=120)
            
            if res.status_code == 200:
                return res.json().get("response", "Empty response")
            return f"LLM Error: {res.status_code} - {res.text}"
            
        except requests.exceptions.Timeout:
            logger.error("LLM Generation timed out.")
            return "Error: LLM Generation timed out after 120 seconds."
        except Exception as e:
            logger.error(f"LLM Connection Error: {e}")
            return f"Connection Error: {e}"