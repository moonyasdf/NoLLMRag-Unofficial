import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MusiqueLoader:
    """Especializado en cargar el dataset MuSiQue."""
    
    @staticmethod
    def load_corpus(file_path: str) -> List[str]:
        """Carga el corpus para indexación (Título + Texto)."""
        logger.info(f"Loading MuSiQue corpus from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Unimos Título y Texto para preservar el contexto en el grafo
        return [f"Title: {item['title']}\nContent: {item['text']}" for item in data]

    @staticmethod
    def load_questions(file_path: str) -> List[Dict[str, Any]]:
        """Carga las preguntas y sus documentos de soporte (gold docs)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)