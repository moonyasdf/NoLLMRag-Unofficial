import sys
import os
import logging
import json
import re
from typing import List, Dict, Any

# Fix para importaciones en Colab/Entornos con path raíz variable
current_path = os.getcwd()
if current_path not in sys.path:
    sys.path.append(current_path)

from src.pipeline import NoLLMRAGPipeline
from reproduction.musique_loader import MusiqueLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MusiqueEvaluator:
    def __init__(self, pipeline: NoLLMRAGPipeline):
        self.pipeline = pipeline

    def _normalize(self, text: str) -> str:
        """Elimina espacios, caracteres especiales y normaliza a minúsculas para matching robusto."""
        return re.sub(r'[^a-z0-9]', '', text.lower())

    def run(self, questions_path: str, k_values: List[int] = [1, 5, 10, 30]):
        if not os.path.exists(questions_path):
            logger.error(f"Questions file not found: {questions_path}")
            return

        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        results = []
        logger.info(f"Starting evaluation of {len(questions)} queries...")

        for i, q_item in enumerate(questions):
            query = q_item['question']
            gold_chunks = [p['paragraph_text'] for p in q_item['paragraphs'] if p['is_supporting']]
            
            # Retrieval puro
            retrieved = self.pipeline.retrieve_only(query)
            
            metrics = self._calculate_metrics(gold_chunks, retrieved, k_values)
            results.append(metrics)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i+1}/{len(questions)}...")

        self._print_summary(results, k_values)

    def _calculate_metrics(self, gold_chunks: List[str], retrieved: List[str], k_values: List[int]) -> Dict:
        query_res = {}
        # Normalizamos los gold chunks de antemano
        gold_normalized = [self._normalize(g[:100]) for g in gold_chunks]
        
        for k in k_values:
            top_k = retrieved[:k]
            top_k_normalized = [self._normalize(r) for r in top_k]
            
            hits = 0
            for gn in gold_normalized:
                if any(gn in rn for rn in top_k_normalized):
                    hits += 1
            
            recall = hits / len(gold_chunks) if gold_chunks else 0
            noise_ratio = (len(top_k) - hits) / len(top_k) if top_k else 1.0
            
            query_res[f"Recall@{k}"] = recall
            query_res[f"Noise@{k}"] = noise_ratio
            query_res[f"FullSuccess@{k}"] = 1.0 if hits == len(gold_chunks) else 0.0
            
        return query_res

    def _print_summary(self, results: List[Dict], k_values: List[int]):
        num_q = len(results)
        print("\n" + "="*50)
        print(f"MUSIQUE EVALUATION SUMMARY ({num_q} Queries)")
        print("="*50)
        
        for k in k_values:
            avg_recall = sum(r[f"Recall@{k}"] for r in results) / num_q
            avg_noise = sum(r[f"Noise@{k}"] for r in results) / num_q
            avg_success = sum(r[f"FullSuccess@{k}"] for r in results) / num_q
            
            print(f"K = {k}:")
            print(f"  - Avg Recall:      {avg_recall:.2%}")
            print(f"  - Avg Noise Ratio: {avg_noise:.2%}")
            print(f"  - Full Hop Success: {avg_success:.2%}")
            print("-" * 20)

if __name__ == "__main__":
    # Ajusta estas rutas a tu entorno de Colab
    CORPUS = "./data/musique_corpus.json"
    QUESTIONS = "./data/musique_4hop_questions.json"

    rag = NoLLMRAGPipeline()
    
    if not rag.ge.graph.vcount():
        logger.info("Index empty. Building from corpus...")
        from reproduction.musique_loader import MusiqueLoader
        docs = MusiqueLoader.load_corpus(CORPUS)
        rag.index(docs)
    
    evaluator = MusiqueEvaluator(rag)
    evaluator.run(QUESTIONS)