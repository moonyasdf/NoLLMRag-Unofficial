import sys
import os
import logging
from typing import List, Dict

# Añadir el directorio raíz al path para poder importar 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import NoLLMRAGPipeline
from reproduction.musique_loader import MusiqueLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MusiqueEvaluator:
    def __init__(self, pipeline: NoLLMRAGPipeline):
        self.pipeline = pipeline

    def run(self, questions_path: str, k_values: List[int] = [1, 5, 10, 30]):
        questions = MusiqueLoader.load_questions(questions_path)
        
        results = []
        logger.info(f"Starting evaluation of {len(questions)} queries...")

        for i, q_item in enumerate(questions):
            query = q_item['question']
            # Chunks de oro (los que contienen la respuesta)
            gold_chunks = [p['paragraph_text'] for p in q_item['paragraphs'] if p['is_supporting']]
            
            # Ejecutar retrieval puro (sin LLM)
            retrieved = self.pipeline.retrieve_only(query)
            
            # Métricas por query
            metrics = self._calculate_metrics(gold_chunks, retrieved, k_values)
            results.append(metrics)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i+1}/{len(questions)}...")

        self._print_summary(results, k_values)

    def _calculate_metrics(self, gold_chunks: List[str], retrieved: List[str], k_values: List[int]) -> Dict:
        query_res = {}
        for k in k_values:
            top_k = retrieved[:k]
            hits = 0
            for gold in gold_chunks:
                # Normalización mínima para comparación
                gold_clean = gold[:100].lower()
                if any(gold_clean in r.lower() for r in top_k):
                    hits += 1
            
            # Recall: ¿Cuántos de los necesarios encontramos?
            recall = hits / len(gold_chunks) if gold_chunks else 0
            
            # Noise Ratio: ¿Cuánta basura hay en lo recuperado?
            # (Documentos recuperados que no son útiles / Total recuperado)
            noise_ratio = (len(top_k) - hits) / len(top_k) if top_k else 1.0
            
            query_res[f"Recall@{k}"] = recall
            query_res[f"Noise@{k}"] = noise_ratio
            
            # Métrica Multi-hop Crítica: ¿Encontramos TODOS los saltos?
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
            print(f"  - Full Hop Success: {avg_success:.2%}") # ¿Encontró los 4 docs?
            print("-" * 20)

if __name__ == "__main__":
    # Rutas relativas
    CORPUS = "./data/musique_corpus.json"
    QUESTIONS = "./data/musique_4hop_questions.json"

    rag = NoLLMRAGPipeline()
    
    # 1. Indexación (Si el grafo está vacío)
    if not rag.ge.graph.vcount():
        logger.info("Index empty. Building from corpus...")
        docs = MusiqueLoader.load_corpus(CORPUS)
        rag.index(docs)
    
    # 2. Evaluación
    evaluator = MusiqueEvaluator(rag)
    evaluator.run(QUESTIONS)