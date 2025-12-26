# NoLLMRAG: Unofficial Optimized Implementation & Benchmarking

This repository contains a high-performance implementation of the NoLLMRAG framework, optimized for large-scale datasets and evaluated against the MuSiQue 4-hop reasoning benchmark.

## Project Scope
The goal was to implement the 3-layer heterogeneous graph index described in the NoLLMRAG paper and evaluate its retrieval capabilities in deep multi-hop scenarios (4 jumps) without using LLMs for indexing or retrieval logic.

## Key Implementation Details & Optimizations
- **Graph Construction**: Vectorized using NumPy and `igraph` to handle millions of edges ($E_{TT}$) in seconds.
- **NLP Pipeline**: Parallelized SpaCy processing using $N-1$ CPU cores.
- **Algorithmic Fixes**:
    - Implemented strict context continuity logic from Appendix B.3 (Algorithm 1).
    - Removed redundant outer logarithm in the Importance Score (IS) formula to improve keyword discrimination.
    - Switched from Lemma-based to Text-based tokenization to preserve entity integrity.
- **Search Logic**: Optimized `VectorStore` to reconstruct vectors directly from FAISS memory, bypassing redundant CPU encoding during inference.

## Evaluation Methodology
- **Dataset**: MuSiQue (2,149 documents, 405 4-hop queries).
- **Environment**: Google Colab (T4 GPU + Local SSD).
- **Metrics**: 
    - **Recall@30**: Percentage of ground-truth documents retrieved.
    - **Noise Ratio**: Percentage of irrelevant documents in the top-k results.
    - **Full Hop Success**: Percentage of queries where all 4 required supporting documents were found.

## Benchmarking Results

| Experiment | Configuration | Recall@30 | Noise Ratio | Full Success |
|:---|:---|:---:|:---:|:---:|
| **Baseline** | Lemma mode, $\tau=0.5$, Intersection | 2.65% | 99.20% | 0.00% |
| **Optimized** | Text mode, $\tau=0.2$, Union mode | 31.91% | 95.57% | 0.25% |
| **Guided** | Question Decomposition (Sub-queries) | **37.10%** | **94.02%** | **0.99%** |

## Critical Observations
1. **Multi-hop Bottleneck**: The static co-occurrence graph is insufficient for 4-hop reasoning. The system excels at finding documents sharing explicit keywords but fails when a bridge document is required to discover the next entity in the logical chain.
2. **Precision vs. Recall**: Lowering the $\tau$ threshold and using the "Union" retrieval mode significantly improved Recall, but at the cost of an extremely high Noise Ratio (>94%).
3. **Decomposition Impact**: Segmenting the query into sub-questions improved Recall by ~5%, confirming that NoLLMRAG's primary weakness is query-to-graph mapping for complex, multi-entity sentences.

## Conclusion
The NoLLMRAG architecture is a robust and efficient "Keyword Discovery Engine." However, for deep reasoning tasks like MuSiQue 4-hop, it requires an iterative or guided retrieval mechanism to overcome the lack of semantic bridging in static graphs.