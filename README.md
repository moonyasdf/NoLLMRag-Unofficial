# NoLLMRAG: Unofficial Optimized Implementation & Benchmarking

This repository contains an unofficial implementation of the [NoLLMRAG framework](https://openreview.net/forum?id=KIUOtEKzzN), optimized for large-scale datasets and evaluated against the MuSiQue 4-hop reasoning benchmark.

## Implementation Overview

This version focuses on performance and scalability, addressing bottlenecks found in the original algorithmic description when applied to multi-thousand document corpora.

### Technical Adjustments
- **Vectorized Graph Construction**: Utilizes NumPy and `igraph` for edge generation. Document-to-token sequences ($E_{TT}$) are processed using matrix operations instead of iterative Python loops.
- **Parallel NLP**: Pre-processing via SpaCy uses multi-core CPU parallelism ($N-1$ cores) to handle tokenization and lemmatization.
- **Statistical Refinement**: The Importance Score ($IS$) was calculated by removing the outer natural logarithm present in the paper's Equation 2. This was done to prevent excessive value compression and preserve keyword discrimination.
- **Retrieval Optimization**: Candidate ranking in the `VectorStore` reconstructs vectors directly from FAISS memory buffers, eliminating redundant neural network inference during the retrieval phase.

## Experimental Setup & Reproducibilidad

The repository includes a configuration system in `src/config.py` to replicate different retrieval behaviors.

### Configuration Flags
To replicate the benchmarks below, adjust the following parameters:

1.  **Baseline (Strict Paper Implementation)**:
    - `TOKEN_MODE = "lemma"`
    - `RETRIEVAL_MODE = "intersection"`
    - `KEYWORD_TAU = 0.5`
2.  **Optimized (Entity Preservation)**:
    - `TOKEN_MODE = "text"`
    - `RETRIEVAL_MODE = "union"`
    - `KEYWORD_TAU = 0.2`

### Evaluation
- **Dataset**: MuSiQue 4-hop (2,149 documents, 405 queries).
- **Environment**: Google Colab (T4 GPU for embeddings, Local SSD for index storage).
- **Metrics**: Recall@30 (Retrieval success), Noise Ratio (Precision failure), and Full Hop Success (Chain completion).

## Benchmarking Results

The following data reflects the performance of the engine across different configurations:

| Experiment | Configuration | Recall@30 | Noise Ratio | Full Hop Success |
|:---|:---|:---:|:---:|:---:|
| **Baseline** | Lemma mode, $\tau=0.5$, Intersection | 2.65% | 99.20% | 0.00% |
| **Optimized** | Text mode, $\tau=0.2$, Union mode | 31.91% | 95.57% | 0.25% |
| **Guided** | Question Decomposition (Sub-queries) | **37.10%** | **94.02%** | **0.99%** |

## Critical Technical Observations

1.  **Semantic Corruption in Lemmatization**: The default lemmatization suggested in the paper frequently deforms entities (e.g., proper nouns like "Guangling"). This leads to a total failure in token-to-graph mapping, explaining the low baseline performance.
2.  **Graph Topology Limitations**: Static co-occurrence graphs do not inherently encode the relational "jumps" required for 4-hop reasoning. If two entities do not appear within the same 100-token chunk anywhere in the corpus, no logical path exists between them in the graph.
3.  **Intersection vs. Union**: The strict intersection logic (Algorithm 1) is too restrictive for deep multi-hop chains. Using a union-based approach significantly increases Recall, although it introduces a high volume of irrelevant context (Noise Ratio > 94%).
4.  **Query Complexity**: Results from the **Guided** experiment confirm that the framework performs better when queries are segmented. NoLLMRAG's keyword extraction struggle to identify all necessary entities from a single complex, multi-layered sentence.

## Repository Structure
```text
.
├── src/
│   ├── config.py          # Centralized hyperparameter management
│   ├── graph_engine.py    # Vectorized graph logic (Union/Intersection)
│   ├── pipeline.py        # Orchestration and retrieval methods
│   ├── text_processor.py  # Parallelized SpaCy processing
│   └── vector_store.py    # Optimized FAISS indexing
├── reproduction/
│   ├── evaluate_musique.py            # Standard benchmark script
│   └── evaluate_musique_decomposed.py # Guided retrieval script
└── main.py                # Minimal usage example
```

## Usage
1. Install dependencies: `pip install -r requirements.txt`.
2. Download the SpaCy model: `python -m spacy download en_core_web_lg`.
3. Configure `TOKEN_MODE` and `RETRIEVAL_MODE` in `src/config.py`.
4. Run the evaluation: `python -u reproduction/evaluate_musique.py`.
