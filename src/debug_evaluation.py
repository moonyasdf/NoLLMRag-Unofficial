import sys
import os
import json
import re
from src.pipeline import NoLLMRAGPipeline
from src.config import Config

def clean_text(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

def run_forensic_v2(pipeline, qa_path):
    with open(qa_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"\n{'='*60}\nANALISIS FORENSE V2 - NoLLMRAG\n{'='*60}")
    print(f"CONFIG TAU: {Config.KEYWORD_TAU}")

    for i in range(3): # Analizar 3 preguntas
        item = questions[i]
        query = item['question']
        gold_chunks = [p['paragraph_text'] for p in item['paragraphs'] if p['is_supporting']]
        
        print(f"\nQUERY {i+1}: {query}")
        
        # 1. TEST DE TEXT_PROCESSOR
        tokens = pipeline.tp.process_chunk_sequence(query)
        print(f"TOKENS (TEXT MODE): {tokens}")

        # 2. TEST DE NAIVE RAG
        retrieved = pipeline.retrieve_naive(query, k=30)
        
        print(f"BUSCANDO {len(gold_chunks)} CHUNKS...")
        for j, gold in enumerate(gold_chunks):
            gn = clean_text(gold[:100])
            found = False
            best_overlap = 0
            
            for r in retrieved:
                rn = clean_text(r)
                # Calculamos coincidencia
                if gn in rn:
                    found = True
                    break
            
            status = "✅ ENCONTRADO" if found else "❌ NO ENCONTRADO"
            print(f"  - Gold Chunk {j+1}: {status}")
            if not found:
                 print(f"    [DEBUG] Esperaba algo como: {gold[:60]}...")
                 print(f"    [DEBUG] El mejor match del retriever fue: {retrieved[0][:60]}...")

        # 3. TEST DE KEYWORDS
        candidate_tids = []
        for t in tokens:
            idx, _ = pipeline.vs.search_tokens(t, top_k=1)
            if idx[0] != -1:
                candidate_tids.append(pipeline.ge.token_start_id + int(idx[0]))
        
        stats = pipeline.ge.calculate_importance_scores(candidate_tids)
        if stats:
            max_s = max(stats.values())
            passed = [pipeline.ge.graph.vs[tid]['label'] for tid, s in stats.items() if s > Config.KEYWORD_TAU * max_s]
            print(f"KEYWORDS QUE PASARON TAU: {passed}")