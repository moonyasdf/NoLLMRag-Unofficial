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
        Construcción de alto rendimiento del grafo de 3 capas (Docs, Chunks, Tokens).
        Optimizado con NumPy y procesamiento por lotes.
        """
        if not documents:
            raise ValueError("La lista de documentos está vacía.")

        logger.info(f"Iniciando indexación de {len(documents)} documentos...")
        
        corpus_data = []
        unique_tokens = set()
        temp_chunk_list = []
        
        logger.info("Fase 1: Segmentación de texto...")
        for doc_id, doc_text in enumerate(documents):
            chunks = processor.segment_text(doc_text)
            for chunk in chunks:
                temp_chunk_list.append((doc_id, chunk))
        
        chunk_texts = [x[1] for x in temp_chunk_list]
        total_chunks = len(chunk_texts)
        
        logger.info(f"Fase 2: Procesando {total_chunks} chunks en paralelo (batch_size={Config.BATCH_SIZE})...")
        
        processed_chunks = list(processor.nlp.pipe(
            chunk_texts, 
            batch_size=Config.BATCH_SIZE, 
            n_process=Config.NUM_PROCESSES
        ))
        
        for i, spacy_doc in enumerate(processed_chunks):
            doc_id, original_text = temp_chunk_list[i]
            # Extraemos tokens (NoLLMRAG usa lemas en el paper, pero aquí usamos texto por robustez)
            tokens_seq = [
                t.text for t in spacy_doc 
                if t.is_alpha and not t.is_stop and len(t.text) > 1
            ]
            corpus_data.append({
                'doc_id': doc_id,
                'chunk_text': original_text,
                'tokens': tokens_seq
            })
            unique_tokens.update(tokens_seq)

        sorted_tokens = sorted(list(unique_tokens))
        
        num_docs = len(documents)
        num_chunks = len(corpus_data)
        num_tokens = len(sorted_tokens)
        
        logger.info(f"Nodos creados: {num_docs} Docs, {num_chunks} Chunks, {num_tokens} Tokens")
        
        chunk_start_id = num_docs
        token_start_id = num_docs + num_chunks
        self._token_start_id = token_start_id
        
        self.vocab_map = {t: (token_start_id + i) for i, t in enumerate(sorted_tokens)}
        
        logger.info("Fase 3: Generando aristas (E_DC, E_CT, E_TT)...")
        
        edges_dc = [] 
        edges_ct = [] 
        edges_tt = [] 
        
        self.chunk_data = {}
        doc_to_chunks_map = defaultdict(list)
        
        for c_idx, data in enumerate(corpus_data):
            c_real_id = chunk_start_id + c_idx
            self.chunk_data[c_real_id] = data['chunk_text']
            
            # E_DC: Documento -> Chunk
            edges_dc.append((data['doc_id'], c_real_id))
            
            t_ids = [self.vocab_map[t] for t in data['tokens']]
            doc_to_chunks_map[data['doc_id']].append((c_real_id, t_ids))
            
            if not t_ids: continue
            
            # E_CT: Chunk <-> Token
            for tid in set(t_ids):
                edges_ct.append((c_real_id, tid))
        
        # E_TT: Token -> Token (Continuidad sintáctica)
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

        # Construcción final en igraph
        self.graph = ig.Graph(directed=True)
        self.graph.add_vertices(num_docs + num_chunks + num_tokens)
        
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
        
        logger.info(f"Grafo construido. Total aristas: {self.graph.ecount()}")
        return sorted_tokens, [self.chunk_data[k] for k in sorted(self.chunk_data.keys())]

    def calculate_importance_scores(self, query_token_ids: List[int]) -> Dict[int, float]:
        """
        Calcula IS (Importance Score) basado en IGTF, ICF y IDF.
        Optimización: Eliminado el logaritmo exterior para mejorar la discriminación.
        """
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
                if etype == 1: # E_CT
                    neighbor = edge.source if edge.target == tid else edge.target
                    if self.graph.vs[neighbor]["type"] == "chunk":
                        connected_chunks_ids.add(neighbor)
                elif etype == 2: # E_TT
                    if edge.target == tid: in_degree_TT += 1
                    if edge.source == tid: out_degree_TT += 1

            # ICF: Rareza en chunks
            icf_val = np.log(total_chunks / (len(connected_chunks_ids) + 1) + 1)
            
            # IDF: Rareza en documentos
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
            
            # IGTF: Rareza global en secuencias
            max_deg = max(in_degree_TT, out_degree_TT)
            if max_deg == 0: max_deg = 1
            igtf_val = np.log((self.total_ETT / max_deg) + 1)
            
            # Score final sin el segundo logaritmo del paper
            scores[tid] = igtf_val * icf_val * idf_val 
            
        return scores

    def extract_keywords_and_cluster(self, query_token_ids: List[int]) -> Set[int]:
        """
        Identifica keywords y recupera chunks. 
        MODO UNIÓN: Recupera todos los chunks que contienen las keywords importantes.
        Este modo es más robusto para 4-hops que la intersección estricta del paper.
        """
        stats = self.calculate_importance_scores(query_token_ids)
        if not stats: return set()
        
        max_is = max(stats.values())
        # Selección de keywords basada en el umbral TAU (configurado en 0.2)
        keywords = [tid for tid, score in stats.items() if score > Config.KEYWORD_TAU * max_is]
        
        if not keywords: 
             # Fallback: si nada pasa el umbral, tomar los top 5
             sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
             keywords = [x[0] for x in sorted_stats[:5]]
             if not keywords: return set()

        logger.debug(f"Keywords seleccionadas para búsqueda: {[self.graph.vs[k]['label'] for k in keywords]}")

        # Recolectar vecinos (chunks conectados a estas keywords)
        retrieved_chunk_ids = set()
        for k in keywords:
            inc = self.graph.incident(k, mode="all")
            for eid in inc:
                if self.graph.es[eid]["type"] == 1: # E_CT
                    edge = self.graph.es[eid]
                    neighbor = edge.source if edge.target == k else edge.target
                    if self.graph.vs[neighbor]["type"] == "chunk":
                        retrieved_chunk_ids.add(neighbor)
            
        return retrieved_chunk_ids

    def save(self):
        logger.info(f"Guardando grafo en {Config.GRAPH_PATH}")
        with open(Config.GRAPH_PATH, 'wb') as f:
            pickle.dump(self.graph, f)
        meta = {
            'vocab': self.vocab_map, 
            'chunks': self.chunk_data, 
            'total_ETT': self.total_ETT, 
            'token_start_id': self._token_start_id
        }
        with open(Config.METADATA_PATH, 'wb') as f:
            pickle.dump(meta, f)
            
    def load(self) -> bool:
        if os.path.exists(Config.GRAPH_PATH):
            logger.info("Cargando grafo desde disco...")
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