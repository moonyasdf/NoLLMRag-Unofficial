import spacy
import logging
from typing import List
from .config import Config

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        # We enforce CPU usage here. 
        logger.info(f"Loading spaCy model: {Config.SPACY_MODEL} (CPU Mode)...")
        try:
            # Disable components not needed for lemmatization/tokenization
            self.nlp = spacy.load(Config.SPACY_MODEL, disable=["ner", "parser", "textcat"])
        except OSError:
            logger.critical(f"Model not found. Please run: python -m spacy download {Config.SPACY_MODEL}")
            raise RuntimeError(f"Spacy model {Config.SPACY_MODEL} not found.")

    def segment_text(self, text: str) -> List[str]:
        """Splits text into non-overlapping chunks."""
        # nlp.make_doc is faster than nlp() as it skips the pipeline
        doc = self.nlp.make_doc(text)
        tokens = [t.text for t in doc]
        chunks = []
        for i in range(0, len(tokens), Config.CHUNK_SIZE):
            chunk_tokens = tokens[i : i + Config.CHUNK_SIZE]
            chunks.append(" ".join(chunk_tokens))
        return chunks

    def process_chunk_sequence(self, text: str) -> List[str]:
        """Used for query processing (single string). Returns lemma sequence."""
        if not text.strip():
            return []
        doc = self.nlp(text.lower())
        seq = []
        for token in doc:
            if token.is_alpha and not token.is_stop and len(token.text) > 1:
                seq.append(token.lemma_)
        return seq