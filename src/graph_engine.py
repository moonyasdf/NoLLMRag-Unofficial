import igraph as ig
import leidenalg
import numpy as np
import pickle
import os
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any
from .config import Config

# Configurar logger
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
        """
        High-Performance Graph Construction using CPU Parallelism & NumPy.
        """
        if not documents:
            raise ValueError("Document list is empty. Cannot build graph.")

        logger.info(f"Starting indexing for {len(documents)} documents...")
        
        corpus_data = []
        unique_tokens = set()
        temp_chunk_list = []
        
        logger.info("Phase 1: Segmentation...")
        for doc_id, doc_text in enumerate(documents):
            chunks = processor.segment_text(doc_text)
            for chunk in chunks:
                temp_chunk_list.append((doc_id, chunk))
        
        chunk_texts = [x[1] for x in temp_chunk_list]
        total_chunks = len(chunk_texts)
        
        # --- PARALLEL CPU PROCESSING ---
        logger.info(f"Phase 2: Tokenizing {total_chunks} chunks using {Config.NUM_PROCESSES} cores...")
        
        # FIX: Usando Config.BATCH_SIZE
        processed_chunks = list(processor.nlp.pipe(
            chunk_texts, 
            batch_size=Config.BATCH_SIZE, 
            n_process=Config.NUM_PROCESSES
        ))
        
        for i, spacy_doc in enumerate(processed_chunks):
            doc_id, original_text = temp_chunk_list[i]
            tokens_seq = [
                t.lemma_ for t in spacy_doc 
                if t.is_alpha and not t.is_stop and len(t.text) > 1
            ]
            corpus_data.append({
                'doc_id': doc_id,
                'chunk_text': original_text,
                'tokens': tokens_seq
            })
            unique_tokens.update(tokens_seq)

        sorted_tokens = sorted(list(unique_tokens))
        
        # --- Vectorized Node Mapping ---
        num_docs = len(documents)
        num_chunks = len(corpus_data)
        num_tokens = len(sorted_tokens)
        
        logger.info(f"Nodes: {num_docs} Docs, {num_chunks} Chunks, {num_tokens} Tokens")
        
        chunk_start_id = num_docs
        token_start_id = num_docs + num_chunks
        self._token_start_id = token_start_id
        
        self.vocab_map = {t: (token_start_id + i) for i, t in enumerate(sorted_tokens)}
        
        # --- Vectorized Edge Generation ---
        logger.info("Phase 3: Generating edges (Vectorized)...")
        
        edges_dc = [] 
        edges_ct = [] 
        edges_tt = [] 
        
        self.chunk_data = {}
        doc_to_chunks_map = defaultdict(list)
        
        for c_idx, data in enumerate(corpus_data):
            c_real_id = chunk_start_id + c_idx
            self.chunk_data[c_real_id] = data['chunk_text']
            
            # E_DC
            edges_dc.append((data['doc_id'], c_real_id))
            
            t_ids = [self.vocab_map[t] for t in data['tokens']]
            doc_to_chunks_map[data['doc_id']].append((c_real_id, t_ids))
            
            if not t_ids: continue
            
            # E_CT
            for tid in set(t_ids):
                edges_ct.append((c_real_id, tid))
        
        # E_TT Generation
        for doc_id in range(num_docs):
            chunks = doc_to_chunks_map[doc_id]
            if not chunks: continue
            
            full_doc_sequence = []
            for _, t_ids in chunks:
                full_doc_sequence.extend(t_ids)
            
            if len(full_doc_sequence) < 2: continue
            
            arr = np.array(full_doc_sequence, dtype=np.int32)
            sources = arr[:-1]
            targets = arr[1:]
            pairs = np.column_stack((sources, targets))
            edges_tt.extend(map(tuple, pairs))

        # --- Build Graph ---
        self.graph = ig.Graph(directed=True)
        self.graph.add_vertices(num_docs + num_chunks + num_tokens)
        
        # Atributos en lote
        self.graph.vs[0:num_docs]["type"] = "document"
        self.graph.vs[chunk_start_id:token_start_id]["type"] = "chunk"
        self.graph.vs[token_start_id:]["type"] = "token"
        self.graph.vs[token_start_id:]["label"] = sorted_tokens
        
        count_dc, count_ct, count_tt = len(edges_dc), len(edges_ct), len(edges_tt)
        self.graph.add_edges(edges_dc + edges_ct + edges_tt)
        
        types = np.concatenate([
            np.zeros(count_dc, dtype=int),
            np.ones(count_ct, dtype=int),
            np.full(count_tt, 2, dtype=int)
        ])
        self.graph.es["type"] = types
        self.total_ETT = count_tt
        
        logger.info(f"Graph Built. Total Edges: {self.graph.ecount()}")
        return sorted_tokens, [self.chunk_data[k] for k in sorted(self.chunk_data.keys())]

    def calculate_importance_scores(self, query_token_ids: List[int]) -> Dict[int, float]:
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
                etype = edge["type"]
                if etype == 1: 
                    neighbor = edge.source if edge.target == tid else edge.target
                    if self.graph.vs[neighbor]["type"] == "chunk":
                        connected_chunks_ids.add(neighbor)
                elif etype == 2:
                    if edge.target == tid: in_degree_TT += 1
                    if edge.source == tid: out_degree_TT += 1

            icf_val = np.log(total_chunks / (len(connected_chunks_ids) + 1) + 1)
            
            connected_docs = set()
            if connected_chunks_ids:
                for cid in connected_chunks_ids:
                    chunk_edges = self.graph.incident(cid, mode="all")
                    for ceid in chunk_edges:
                         if self.graph.es[ceid]["type"] == 0:
                             cedge = self.graph.es[ceid]
                             doc_node = cedge.source if cedge.target == cid else cedge.target
                             connected_docs.add(doc_node)

            idf_val = np.log(total_docs / (len(connected_docs) + 1) + 1)
            
            max_deg = max(in_degree_TT, out_degree_TT)
            if max_deg == 0: max_deg = 1
            igtf_val = np.log((self.total_ETT / max_deg) + 1)
            
            scores[tid] = igtf_val * icf_val * idf_val 
            
        return scores

    def extract_keywords_and_cluster(self, query_token_ids: List[int]) -> Set[int]:
        stats = self.calculate_importance_scores(query_token_ids)
        if not stats: return set()
        
        max_is = max(stats.values())
        keywords = [tid for tid, score in stats.items() if score > Config.KEYWORD_TAU * max_is]
        
        if not keywords: 
             sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
             keywords = [x[0] for x in sorted_stats[:5]]
             if not keywords: return set()

        # Build G_co (Co-occurrence)
        k_neighbors = {}
        for k in keywords:
            inc = self.graph.incident(k, mode="all")
            chunks = set()
            for eid in inc:
                if self.graph.es[eid]["type"] == 1:
                    edge = self.graph.es[eid]
                    node = edge.source if edge.target == k else edge.target
                    chunks.add(node)
            k_neighbors[k] = chunks

        edges = []
        weights = []
        for i in range(len(keywords)):
            for j in range(i+1, len(keywords)):
                k1, k2 = keywords[i], keywords[j]
                intersection_size = len(k_neighbors[k1].intersection(k_neighbors[k2]))
                if intersection_size > 0:
                    edges.append((i, j))
                    weights.append(intersection_size)
                    
        g_co = ig.Graph(len(keywords))
        g_co.add_edges(edges)
        g_co.vs["original_id"] = keywords
        if weights:
            g_co.es["weight"] = weights
            
        # Clustering
        clusters = []
        if g_co.vcount() > 0:
            if g_co.ecount() > 0:
                partition = leidenalg.find_partition(
                    g_co, leidenalg.ModularityVertexPartition, weights=weights, n_iterations=-1
                )
                clusters = partition
            else:
                clusters = [[i] for i in range(len(keywords))]
        else:
            return set()

        # Retrieval Algorithm (Appendix B.3)
        retrieved_chunk_ids = set()
        
        for cluster_indices in clusters:
            if weights:
                cluster_indices.sort(key=lambda idx: g_co.strength(idx, weights=weights), reverse=True)
            
            cluster_tids = [g_co.vs[i]["original_id"] for i in cluster_indices]
            if not cluster_tids: continue
            
            s_curr = k_neighbors[cluster_tids[0]]
            s_inter = set() 
            union_prev_results = set() 
            
            for i in range(1, len(cluster_tids)):
                next_chunks = k_neighbors[cluster_tids[i]]
                intersection = s_curr.intersection(next_chunks)
                
                if intersection:
                    s_curr = intersection 
                else:
                    overlap = next_chunks.intersection(union_prev_results)
                    if overlap:
                        s_inter.update(s_curr) 
                        union_prev_results.update(s_curr)
                        s_curr = overlap       
                    else:
                        s_inter.update(s_curr)
                        union_prev_results.update(s_curr)
                        s_curr = next_chunks   
            
            s_inter.update(s_curr)
            retrieved_chunk_ids.update(s_inter)
            
        return retrieved_chunk_ids

    def save(self):
        logger.info(f"Saving graph to {Config.GRAPH_PATH}")
        with open(Config.GRAPH_PATH, 'wb') as f:
            pickle.dump(self.graph, f)
        meta = {'vocab': self.vocab_map, 'chunks': self.chunk_data, 'total_ETT': self.total_ETT, 'token_start_id': self._token_start_id}
        with open(Config.METADATA_PATH, 'wb') as f:
            pickle.dump(meta, f)
            
    def load(self) -> bool:
        if os.path.exists(Config.GRAPH_PATH):
            logger.info("Loading graph from disk...")
            with open(Config.GRAPH_PATH, 'rb') as f:
                self.graph = pickle.load(f)
            with open(Config.METADATA_PATH, 'rb') as f:
                data = pickle.load(f)
                self.vocab_map = data['vocab']
                self.chunk_data = data['chunks']
                self.total_ETT = data.get('total_ETT', 0)
                self._token_start_id = data.get('token_start_id', 0)
            return True
        return False