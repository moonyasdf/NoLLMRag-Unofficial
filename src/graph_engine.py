import igraph as ig
import leidenalg
import numpy as np
import pickle
import os
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any
from .config import Config

logger = logging.getLogger(__name__)

class GraphEngine:
    def __init__(self):
        self.graph = ig.Graph(directed=True) 
        self.vocab_map: Dict[str, int] = {} 
        self.chunk_data: Dict[int, str] = {} 
        self.total_ETT: int = 0
        self._token_start_id: int = 0

    @property
    def token_start_id(self) -> int:
        return self._token_start_id

    def build(self, documents: List[str], processor: Any) -> Tuple[List[str], List[str]]:
        """Vectorized construction of the 3-layer heterogeneous graph."""
        if not documents:
            raise ValueError("Document list is empty.")

        logger.info(f"Indexing {len(documents)} documents using mode: {Config.TOKEN_MODE}")
        
        corpus_data = []
        unique_tokens = set()
        temp_chunk_list = []
        
        # Phase 1: Segmentation
        for doc_id, doc_text in enumerate(documents):
            chunks = processor.segment_text(doc_text)
            for chunk in chunks:
                temp_chunk_list.append((doc_id, chunk))
        
        chunk_texts = [x[1] for x in temp_chunk_list]
        
        # Phase 2: Parallel NLP Processing
        processed_chunks = list(processor.nlp.pipe(
            chunk_texts, 
            batch_size=Config.BATCH_SIZE, 
            n_process=Config.NUM_PROCESSES
        ))
        
        for i, spacy_doc in enumerate(processed_chunks):
            doc_id, original_text = temp_chunk_list[i]
            # Use Config-dependent tokenization (text vs lemma)
            if Config.TOKEN_MODE == "lemma":
                tokens_seq = [t.lemma_ for t in spacy_doc if t.is_alpha and not t.is_stop and len(t.text) > 1]
            else:
                tokens_seq = [t.text for t in spacy_doc if t.is_alpha and not t.is_stop and len(t.text) > 1]
                
            corpus_data.append({'doc_id': doc_id, 'chunk_text': original_text, 'tokens': tokens_seq})
            unique_tokens.update(tokens_seq)

        sorted_tokens = sorted(list(unique_tokens))
        num_docs, num_chunks, num_tokens = len(documents), len(corpus_data), len(sorted_tokens)
        
        chunk_start_id = num_docs
        token_start_id = num_docs + num_chunks
        self._token_start_id = token_start_id
        self.vocab_map = {t: (token_start_id + i) for i, t in enumerate(sorted_tokens)}
        
        # Phase 3: Edge Generation
        edges_dc, edges_ct, edges_tt = [], [], []
        doc_to_chunks_map = defaultdict(list)
        
        for c_idx, data in enumerate(corpus_data):
            c_real_id = chunk_start_id + c_idx
            self.chunk_data[c_real_id] = data['chunk_text']
            edges_dc.append((data['doc_id'], c_real_id))
            t_ids = [self.vocab_map[t] for t in data['tokens']]
            doc_to_chunks_map[data['doc_id']].append((c_real_id, t_ids))
            for tid in set(t_ids):
                edges_ct.append((c_real_id, tid))
        
        # Vectorized E_TT
        for doc_id in range(num_docs):
            chunks = doc_to_chunks_map[doc_id]
            if not chunks: continue
            full_seq = []
            for _, t_ids in chunks: full_seq.extend(t_ids)
            if len(full_seq) < 2: continue
            arr = np.array(full_seq, dtype=np.int32)
            pairs = np.column_stack((arr[:-1], arr[1:]))
            edges_tt.extend(map(tuple, pairs))

        # Phase 4: Graph Assembly
        self.graph = ig.Graph(directed=True)
        self.graph.add_vertices(num_docs + num_chunks + num_tokens)
        self.graph.vs[0:num_docs]["type"] = "document"
        self.graph.vs[chunk_start_id:token_start_id]["type"] = "chunk"
        self.graph.vs[token_start_id:]["type"] = "token"
        self.graph.vs[token_start_id:]["label"] = sorted_tokens
        
        self.graph.add_edges(edges_dc + edges_ct + edges_tt)
        types = np.concatenate([np.zeros(len(edges_dc)), np.ones(len(edges_ct)), np.full(len(edges_tt), 2)])
        self.graph.es["type"] = types
        self.total_ETT = len(edges_tt)
        
        logger.info(f"Graph Built: {self.graph.vcount()} nodes, {self.graph.ecount()} edges.")
        return sorted_tokens, [self.chunk_data[k] for k in sorted(self.chunk_data.keys())]

    def calculate_importance_scores(self, query_token_ids: List[int]) -> Dict[int, float]:
        """Calculates token importance scores. Optimized logic (Single Log)."""
        scores = {}
        total_chunks = len(self.chunk_data)
        total_docs = len(self.graph.vs.select(type_eq="document"))
        
        for tid in query_token_ids:
            inc_edges = self.graph.incident(tid, mode="all")
            connected_chunks_ids = set()
            in_degree_TT = 0
            out_degree_TT = 0
            
            for eid in inc_edges:
                edge = self.graph.es[eid]
                if edge["type"] == 1:
                    neighbor = edge.source if edge.target == tid else edge.target
                    if self.graph.vs[neighbor]["type"] == "chunk": connected_chunks_ids.add(neighbor)
                elif edge["type"] == 2:
                    if edge.target == tid: in_degree_TT += 1
                    if edge.source == tid: out_degree_TT += 1

            icf = np.log(total_chunks / (len(connected_chunks_ids) + 1) + 1)
            connected_docs = set()
            for cid in connected_chunks_ids:
                for ceid in self.graph.incident(cid, mode="all"):
                    if self.graph.es[ceid]["type"] == 0:
                        e = self.graph.es[ceid]
                        connected_docs.add(e.source if e.target == cid else e.target)
            idf = np.log(total_docs / (len(connected_docs) + 1) + 1)
            igtf = np.log((self.total_ETT / max(1, max(in_degree_TT, out_degree_TT))) + 1)
            
            # Resulting Importance Score
            scores[tid] = igtf * icf * idf
        return scores

    def extract_keywords_and_cluster(self, query_token_ids: List[int]) -> Set[int]:
        """Retrieves chunks based on selected Config.RETRIEVAL_MODE."""
        stats = self.calculate_importance_scores(query_token_ids)
        if not stats: return set()
        
        max_is = max(stats.values())
        keywords = [tid for tid, score in stats.items() if score > Config.KEYWORD_TAU * max_is]
        if not keywords: keywords = [x[0] for x in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]]

        # --- MODE 1: UNION (Optimized for Multi-hop) ---
        if Config.RETRIEVAL_MODE == "union":
            retrieved = set()
            for k in keywords:
                for eid in self.graph.incident(k, mode="all"):
                    if self.graph.es[eid]["type"] == 1:
                        edge = self.graph.es[eid]
                        node = edge.source if edge.target == k else edge.target
                        if self.graph.vs[node]["type"] == "chunk": retrieved.add(node)
            return retrieved

        # --- MODE 2: INTERSECTION (Strict Paper Algorithm 1) ---
        # 1. Build Co-occurrence Subgraph
        k_neighbors = {}
        for k in keywords:
            nodes = set()
            for eid in self.graph.incident(k, mode="all"):
                if self.graph.es[eid]["type"] == 1:
                    e = self.graph.es[eid]
                    nodes.add(e.source if e.target == k else e.target)
            k_neighbors[k] = nodes

        g_co = ig.Graph(len(keywords))
        weights = []
        for i in range(len(keywords)):
            for j in range(i+1, len(keywords)):
                w = len(k_neighbors[keywords[i]].intersection(k_neighbors[keywords[j]]))
                if w > 0:
                    g_co.add_edge(i, j)
                    weights.append(w)
        
        # 2. Leiden Clustering
        partition = leidenalg.find_partition(g_co, leidenalg.ModularityVertexPartition, weights=weights or None)
        retrieved_final = set()
        for cluster_indices in partition:
            if weights: cluster_indices.sort(key=lambda idx: g_co.strength(idx, weights=weights), reverse=True)
            cluster_tids = [keywords[i] for i in cluster_indices]
            s_curr = k_neighbors[cluster_tids[0]]
            s_inter = set()
            union_prev = set()
            for i in range(1, len(cluster_tids)):
                next_chunks = k_neighbors[cluster_tids[i]]
                intersection = s_curr.intersection(next_chunks)
                if intersection: s_curr = intersection
                else:
                    overlap = next_chunks.intersection(union_prev)
                    if overlap: s_inter.update(s_curr); union_prev.update(s_curr); s_curr = overlap
                    else: s_inter.update(s_curr); union_prev.update(s_curr); s_curr = next_chunks
            s_inter.update(s_curr)
            retrieved_final.update(s_inter)
        return retrieved_final

    def save(self):
        with open(Config.GRAPH_PATH, 'wb') as f: pickle.dump(self.graph, f)
        meta = {'vocab': self.vocab_map, 'chunks': self.chunk_data, 'total_ETT': self.total_ETT, 'token_start_id': self._token_start_id}
        with open(Config.METADATA_PATH, 'wb') as f: pickle.dump(meta, f)
            
    def load(self) -> bool:
        if os.path.exists(Config.GRAPH_PATH):
            with open(Config.GRAPH_PATH, 'rb') as f: self.graph = pickle.load(f)
            with open(Config.METADATA_PATH, 'rb') as f:
                data = pickle.load(f)
                self.vocab_map, self.chunk_data, self.total_ETT = data['vocab'], data['chunks'], data['total_ETT']
                self._token_start_id = data.get('token_start_id', 0)
            return True
        return False
