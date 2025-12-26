import sys
import os
import json
import logging
import re
from typing import List, Dict

# Path setup para Colab local
sys.path.append("/content/NoLLMRAG_Local")

from src.pipeline import NoLLMRAGPipeline
from reproduction.musique_loader import MusiqueLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DecomposedEvaluator:
    def __init__(self, pipeline: NoLLMRAGPipeline):
        self.pipeline = pipeline

    def _normalize(self, text: str) -> str:
        return re.sub(r'[^a-z0-9]', '', text.lower())

    def run(self, questions_path: str):
        # --- VERIFICACIÓN DE SEGURIDAD ---
        if self.pipeline.ge.graph.vcount() == 0:
            logger.error("EL GRAFO ESTÁ VACÍO. Los índices no se cargaron correctamente en /content/NoLLMRAG_Local/indices/")
            return

        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        results = []
        logger.info(f"Iniciando Evaluación Descompuesta sobre {len(questions)} preguntas...")

        for i, q_item in enumerate(questions):
            query = q_item['question']
            # Extraer sub-preguntas
            sub_queries = [sq['question'] for sq in q_item['question_decomposition']]
            gold_chunks = [p['paragraph_text'] for p in q_item['paragraphs'] if p['is_supporting']]
            
            # Retrieval guiado por la descomposición
            retrieved = self.pipeline.retrieve_decomposed(sub_queries, query)
            
            metrics = self._calculate(gold_chunks, retrieved)
            results.append(metrics)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Procesadas {i+1}/{len(questions)} queries...")

        self._summary(results)

    def _calculate(self, gold_chunks, retrieved):
        top_k = retrieved[:30]
        gold_norm = [self._normalize(g[:100]) for g in gold_chunks]
        ret_norm = [self._normalize(r) for r in top_k]
        
        hits = 0
        for gn in gold_norm:
            if any(gn in rn for rn in ret_norm):
                hits += 1
        
        recall = hits / len(gold_chunks) if gold_chunks else 0
        success = 1.0 if hits == len(gold_chunks) else 0.0
        return {"recall": recall, "success": success}

    def _summary(self, results):
        num = len(results)
        if num == 0: return
        avg_recall = sum(r['recall'] for r in results) / num
        avg_success = sum(r['success'] for r in results) / num
        print("\n" + "="*50)
        print("RESULTADOS: NOLLMRAG + DESCOMPOSICIÓN (GUIDED)")
        print("="*50)
        print(f"Recall@30 Promedio:  {avg_recall:.2%}")
        print(f"Full Success@30:      {avg_success:.2%}")
        print("="*50)

if __name__ == "__main__":
    # Inicializar Pipeline
    rag = NoLLMRAGPipeline()
    
    # Comprobación de archivos locales
    evaluator = DecomposedEvaluator(rag)
    # Ajusta la ruta si es necesario
    evaluator.run("./data/musique_4hop_questions.json")