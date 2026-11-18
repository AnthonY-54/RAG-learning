"""
rag_pipeline.py — Stage 3: Interactive RAG with Configurable Advanced Retrieval
"""

import os
import glob
from dotenv import load_dotenv
from groq import Groq

from loader import load_document
from chunkers import recursive_character_chunker
from embedder import init_embedder, get_embeddings
from retrievers import HybridRetriever, CrossEncoderReRanker
from vector_store import create_collection, add_chunks_to_collection, retrieve_dense, retrieve_mmr


def setup_groq() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing in .env")
    return Groq(api_key=api_key)


def main():
    print("=" * 70)
    print(" 🛠️  STAGE 3: ADVANCED RETRIEVAL & RE-RANKING PIPELINE")
    print("=" * 70)

    # 1. Select document
    doc_files = glob.glob("docs/*.txt") + glob.glob("docs/*.pdf")
    if not doc_files:
        print("No documents found in docs/!")
        return

    print("\nAvailable documents:")
    for idx, f in enumerate(doc_files, 1):
        print(f"  [{idx}] {f}")

    doc_choice = input(f"Select document [1-{len(doc_files)}, default=1]: ").strip() or "1"
    selected_doc = doc_files[int(doc_choice) - 1]

    # 2. Load & Chunk
    text, meta = load_document(selected_doc)
    chunks = recursive_character_chunker(text, chunk_size=380, chunk_overlap=40)
    print(f"\n[Prep] Document split into {len(chunks)} chunks.")

    # 3. Models & Indexing
    embedder = init_embedder()
    groq_client = setup_groq()
    reranker = CrossEncoderReRanker()

    print("[Prep] Embedding chunks...")
    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = get_embeddings(embedder, chunk_texts)

    collection = create_collection("rag_stage3_interactive")
    add_chunks_to_collection(collection, chunks, chunk_embeddings, source=meta["source"])

    hybrid_engine = HybridRetriever(chunks, chunk_embeddings)

    # 4. Choose retrieval strategy
    print("\nSelect Retrieval Strategy:")
    print("  1. Dense Cosine (standard top-k vector search)")
    print("  2. MMR (Maximum Marginal Relevance — diversity-aware)")
    print("  3. Sparse BM25 (keyword/exact token search)")
    print("  4. Hybrid Search (Dense + BM25 fused via Reciprocal Rank Fusion)")
    print("  5. Production Pipeline: Hybrid + Cross-Encoder Re-Ranking")

    strat_choice = input("Enter choice [1-5, default=5]: ").strip() or "5"
    top_k = int(input("Number of final chunks to retrieve (top-k) [default 3]: ").strip() or "3")

    print("\n" + "=" * 70)
    print("System Ready! Ask questions.")
    print("Type 'quit' or 'exit' to exit.")
    print("=" * 70 + "\n")

    while True:
        q = input("Your Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not q:
            continue

        q_emb = get_embeddings(embedder, [q])[0]

        # Execute selected strategy
        if strat_choice == "1":
            results = retrieve_dense(collection, q_emb, top_k=top_k)
            strat_name = "Dense (Cosine Top-K)"
        elif strat_choice == "2":
            results = retrieve_mmr(collection, q_emb, top_k=top_k, lambda_param=0.5)
            strat_name = "MMR (Diversity Vectors)"
        elif strat_choice == "3":
            results = hybrid_engine.bm25_retriever.retrieve(q, top_k=top_k)
            strat_name = "Sparse (BM25)"
        elif strat_choice == "4":
            results = hybrid_engine.retrieve(q, q_emb, top_k=top_k)
            strat_name = "Hybrid (BM25 + Dense RRF)"
        else:
            candidates = hybrid_engine.retrieve(q, q_emb, top_k=10)
            results = reranker.rerank(q, candidates, top_k=top_k)
            strat_name = "Hybrid + Cross-Encoder Re-Ranking"

        print(f"\n[Retrieval: {strat_name}] Top {len(results)} chunks:")
        for r in results:
            preview = r["text"].replace("\n", " ").strip()[:90]
            score_info = ""
            for k in ["distance", "score", "rerank_score"]:
                if k in r:
                    score_info += f"{k}={r[k]} "
            print(f"  • Chunk #{r['chunk_id']} | {score_info}| \"{preview}...\"")

        context = "\n\n---\n\n".join([r["text"] for r in results])
        prompt = (
            f"You are a helpful assistant. Answer the user question based strictly on the provided context.\n"
            f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
        )

        res = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        print(f"\n[Answer]:\n{res.choices[0].message.content.strip()}\n")
        print("-" * 70)


if __name__ == "__main__":
    main()
