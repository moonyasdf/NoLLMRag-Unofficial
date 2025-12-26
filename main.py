from src.pipeline import NoLLMRAGPipeline

if __name__ == "__main__":
    rag = NoLLMRAGPipeline()
    
    # 1. Indexing Mode (Comentar si ya está indexado)
    # documents = [
    #     "NoLLMRAG constructs a three-layer heterogeneous graph without LLMs.",
    #     "The Leiden algorithm detects communities in the co-occurrence subgraph.",
    #     "SpaCy is used for lemmatization and tokenization in the preprocessing stage.",
    #     "Graph statistics like IGTF and ICF determine keyword importance."
    # ]
    # rag.index(documents)
    
    # 2. Query Mode
    q = "How does NoLLMRAG select keywords?"
    print(f"\nQuery: {q}")
    answer = rag.query(q)
    print("-" * 30)
    print(f"Answer:\n{answer}")