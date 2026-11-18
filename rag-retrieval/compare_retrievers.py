"""
compare_retrievers.py — Stage 3: Retrieval Strategy Comparison Harness

Compares 5 Retrieval Approaches on the exact same document & query:
  1. Dense Retrieval (Top-K Cosine)
  2. MMR Retrieval (Relevance + Diversity)
  3. Sparse BM25 (Lexical / Exact Matching)
  4. Hybrid Retrieval (BM25 + Dense fused via Reciprocal Rank Fusion)
  5. Re-Ranked Pipeline (Hybrid Candidates -> FlashRank Cross-Encoder)

Shows:
  - Which chunks are retrieved by each method
  - Score metrics (Cosine Distance, BM25 score, RRF score, Cross-Encoder score)
  - Generated answers via Groq for each retrieval method
"""

import os
import sys
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


def run_retrieval_comparison(doc_path: str, query: str):
    print("=" * 85)
    print(" 🔬 STAGE 3: RETRIEVAL QUALITY & RE-RANKING BENCHMARK")
    print("=" * 85)
    print(f"📄 Document : {doc_path}")
    print(f"❓ Query    : '{query}'\n")

    # Step 1: Load and chunk document
    text, meta = load_document(doc_path)
    # Using 500-char chunks to ensure each incident code and its resolution stay intact
    chunks = recursive_character_chunker(text, chunk_size=500, chunk_overlap=60)
    print(f"[Prep] Document split into {len(chunks)} chunks using recursive character chunking.")

    # Step 2: Initialize models
    embedder = init_embedder()
    groq_client = setup_groq()
    reranker = CrossEncoderReRanker()

    # Step 3: Embed chunks & create vector store
    print("\n[Prep] Generating dense embeddings...")
    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = get_embeddings(embedder, chunk_texts)

    collection = create_collection("stage3_bench_col")
    add_chunks_to_collection(collection, chunks, chunk_embeddings, source=meta["source"])

    # Step 4: Initialize Hybrid & Query Embedding
    query_emb = get_embeddings(embedder, [query])[0]
    hybrid_engine = HybridRetriever(chunks, chunk_embeddings)

    TOP_K = 3

    # =========================================================================
    # Strategy 1: Dense (Standard Top-K)
    # =========================================================================
    dense_results = retrieve_dense(collection, query_emb, top_k=TOP_K)

    # =========================================================================
    # Strategy 2: MMR (Relevance + Diversity)
    # =========================================================================
    mmr_results = retrieve_mmr(collection, query_emb, top_k=TOP_K, candidate_fetch_k=10, lambda_param=0.5)

    # =========================================================================
    # Strategy 3: Sparse BM25
    # =========================================================================
    sparse_results = hybrid_engine.bm25_retriever.retrieve(query, top_k=TOP_K)

    # =========================================================================
    # Strategy 4: Hybrid (BM25 + Dense RRF)
    # =========================================================================
    hybrid_results = hybrid_engine.retrieve(query, query_emb, top_k=TOP_K, candidate_pool_size=10)

    # =========================================================================
    # Strategy 5: Re-Ranked (Hybrid Pool -> FlashRank Cross-Encoder)
    # =========================================================================
    hybrid_candidate_pool = hybrid_engine.retrieve(query, query_emb, top_k=8, candidate_pool_size=10)
    reranked_results = reranker.rerank(query, hybrid_candidate_pool, top_k=TOP_K)

    all_methods = [
        ("1. Pure Dense (Cosine Top-K)", dense_results, "distance"),
        ("2. MMR (Diverse Vectors, lambda=0.5)", mmr_results, "distance"),
        ("3. Pure Sparse (BM25 Keyword)", sparse_results, "score"),
        ("4. Hybrid Search (Dense + BM25 via RRF)", hybrid_results, "score"),
        ("5. Re-Ranked (Hybrid Pool -> FlashRank)", reranked_results, "rerank_score"),
    ]

    print("\n" + "=" * 85)
    print(" 📊 DETAILED RETRIEVAL COMPARISON")
    print("=" * 85)

    for name, results, score_key in all_methods:
        print(f"\n--- {name} ---")
        for i, item in enumerate(results, 1):
            score_val = item.get(score_key, "N/A")
            chunk_preview = item["text"].replace("\n", " ").strip()[:100]
            print(f"  [{i}] Chunk #{item['chunk_id']} | {score_key}={score_val} | \"{chunk_preview}...\"")

        # Synthesize answer for this retrieval method
        context = "\n\n---\n\n".join([r["text"] for r in results])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely based strictly on the context above. If unable to answer, say 'Insufficient info':"
        )
        res = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        print(f"  📝 Answer: {res.choices[0].message.content.strip()}")

    print("\n" + "=" * 85)
    print(" 💡 RETRIEVAL SUMMARY TABLE")
    print("=" * 85)
    print(f"{'Method':<42} | {'Retrieved Chunk IDs':<22} | Top Score")
    print("-" * 85)
    for name, results, score_key in all_methods:
        c_ids = str([r["chunk_id"] for r in results])
        top_s = f"{score_key}={results[0].get(score_key, 'N/A')}" if results else "N/A"
        print(f"{name:<42} | {c_ids:<22} | {top_s}")
    print("=" * 85)


if __name__ == "__main__":
    default_doc = "docs/it_system_runbook.txt"
    # Query designed to test exact acronyms/codes where vector search can struggle
    default_query = "What command should I run to resolve ERR_AUTH_9021?"

    doc = sys.argv[1] if len(sys.argv) > 1 else default_doc
    q = sys.argv[2] if len(sys.argv) > 2 else default_query

    run_retrieval_comparison(doc, q)
