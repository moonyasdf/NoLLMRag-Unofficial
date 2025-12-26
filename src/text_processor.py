import spacy
import logging
from typing import List
from .config import Config

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        logger.info(f"Loading spaCy model: {Config.SPACY_MODEL} (CPU Mode)...")
        try:
            # We explicitly exclude heavy components to save RAM
            self.nlp = spacy.load(Config.SPACY_MODEL, disable=["ner", "parser", "textcat"])
        except OSError:
            logger.critical(f"Model not found. Run: python -m spacy download {Config.SPACY_MODEL}")
            raise RuntimeError(f"Spacy model {Config.SPACY_MODEL} not found.")

    def segment_text(self, text: str) -> List[str]:
        """Splits text into non-overlapping chunks based on Config.CHUNK_SIZE."""
        # nlp.make_doc is faster as it skips the neural pipeline
        doc = self.nlp.make_doc(text)
        tokens = [t.text for t in doc]
        chunks = []
        for i in range(0, len(tokens), Config.CHUNK_SIZE):
            chunk_tokens = tokens[i : i + Config.CHUNK_SIZE]
            chunks.append(" ".join(chunk_tokens))
        return chunks

    def process_chunk_sequence(self, text: str) -> List[str]:
        """
        Processes a string into a list of tokens.
        Modes:
        - 'lemma': Returns dictionary forms (Paper original).
        - 'text': Returns raw lowercase text (Preserves entities).
        """
        if not text.strip():
            return []
            
        doc = self.nlp(text.lower())
        
        if Config.TOKEN_MODE == "lemma":
            # Paper implementation: Lemmatization can corrupt names like 'Guangling' -> 'guangle'
            return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop and len(t.text) > 1]
        else:
            # Optimized implementation: Preserve original text for better entity mapping
            return [t.text for t in doc if t.is_alpha and not t.is_stop and len(t.text) > 1]
