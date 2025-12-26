import spacy
import logging
from typing import List
from .config import Config

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        logger.info(f"Loading spaCy model: {Config.SPACY_MODEL} (CPU Mode)...")
        try:
            self.nlp = spacy.load(Config.SPACY_MODEL, disable=["ner", "parser", "textcat"])
        except OSError:
            logger.critical(f"Model not found. Please run: python -m spacy download {Config.SPACY_MODEL}")
            raise RuntimeError(f"Spacy model {Config.SPACY_MODEL} not found.")

    def segment_text(self, text: str) -> List[str]:
        """Splits text into non-overlapping chunks."""
        doc = self.nlp.make_doc(text)
        tokens = [t.text for t in doc]
        chunks = []
        for i in range(0, len(tokens), Config.CHUNK_SIZE):
            chunk_tokens = tokens[i : i + Config.CHUNK_SIZE]
            chunks.append(" ".join(chunk_tokens))
        return chunks

    def process_chunk_sequence(self, text: str) -> List[str]:
        """Used for query processing. Returns original text tokens instead of lemmas to preserve entities."""
        if not text.strip():
            return []
        doc = self.nlp(text.lower())
        seq = []
        for token in doc:
            # Filtramos solo por stopword y alpha, pero preservamos el TEXTO original
            if token.is_alpha and not token.is_stop and len(token.text) > 1:
                seq.append(token.text) # CAMBIO: .text en lugar de .lemma_
        return seq