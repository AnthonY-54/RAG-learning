"""
compare_chunks.py — Stage 2: Chunking Strategy Benchmark & Comparison

Runs the same query against a document chunked with 4 different strategies:
  1. Fixed Character Chunking
  2. Sentence-based Chunking
  3. Recursive Character Chunking
  4. Token-Aware Chunking

Compares:
  - Chunk statistics (count, min/max/avg tokens)
  - Boundary cleanliness (does it cut words/sentences in half?)
  - Retrieval distance (semantic match quality)
  - Generated answer quality
"""

import os
from dotenv import load_dotenv
from groq import Groq

from loader import load_document
from chunkers import (
    fixed_character_chunker,
    sentence_chunker,
    recursive_character_chunker,
    token_chunker,
)
from embedder import init_embedder, get_embeddings
from vector_store import create_collection, add_chunks_to_collection, retrieve


def setup_groq() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing in .env")
    return Groq(api_key=api_key)


def run_comparison(doc_path: str, query: str):
    print("=" * 80)
    print(" 🔬 STAGE 2: CHUNKING STRATEGY COMPARISON HARNESS")
    print("=" * 80)
    print(f"📄 Document : {doc_path}")
    print(f"❓ Query    : '{query}'\n")

    # Step 1: Load text
    text, meta = load_document(doc_path)
    print(f"Document stats: {meta['char_count']} chars, {meta.get('pages', 1)} page(s)\n")

    # Step 2: Initialize models
    embedder = init_embedder()
    groq_client = setup_groq()

    # Step 3: Define strategies to test
    strategies = {
        "1. Fixed Character (500c, 50c overlap)": fixed_character_chunker(text, 500, 50),
        "2. Sentence-Based (3 sentences, 1 overlap)": sentence_chunker(text, 3, 1),
        "3. Recursive Character (500c, 50c overlap)": recursive_character_chunker(text, 500, 50),
        "4. Token-Aware (120 tokens, 20 overlap)": token_chunker(text, 120, 20),
    }

    results = []

    # Step 4: Index and query each strategy independently
    query_emb = get_embeddings(embedder, [query])[0]

    for name, chunks in strategies.items():
        print("-" * 80)
        print(f"Testing Strategy: {name}")

        tokens = [c.token_count for c in chunks]
        chars = [c.char_count for c in chunks]
        avg_tokens = round(sum(tokens) / len(tokens), 1)
        avg_chars = round(sum(chars) / len(chars), 1)

        print(f"  • Total Chunks : {len(chunks)}")
        print(f"  • Token Length : min={min(tokens)}, max={max(tokens)}, avg={avg_tokens}")
        print(f"  • Char Length  : min={min(chars)}, max={max(chars)}, avg={avg_chars}")

        # Index in isolated collection
        col_name = "col_" + name.split(".")[0].strip().replace(" ", "_")
        col = create_collection(col_name)
        chunk_texts = [c.text for c in chunks]
        embs = get_embeddings(embedder, chunk_texts)
        add_chunks_to_collection(col, chunks, embs, source=meta["source"])

        # Retrieve top 2
        top_chunks = retrieve(col, query_emb, top_k=2)
        best = top_chunks[0]
        print(f"  • Top Chunk Match: distance={best['distance']}")
        print(f"  • First 150 chars: \"{best['text'][:150].strip()}...\"")

        # Generate answer with Groq
        context = "\n\n".join([c["text"] for c in top_chunks])
        prompt = (
            f"Based ONLY on the following context, answer the question accurately and concisely.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )

        res = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        answer = res.choices[0].message.content.strip()
        print(f"  • Answer: {answer}")

        results.append({
            "name": name,
            "total_chunks": len(chunks),
            "avg_tokens": avg_tokens,
            "best_distance": best['distance'],
            "answer": answer,
        })

    # Summary table
    print("\n" + "=" * 80)
    print(" 📊 EXECUTIVE SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Strategy':<45} | {'Chunks':<8} | {'Avg Tokens':<11} | {'Top Distance':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<45} | {r['total_chunks']:<8} | {r['avg_tokens']:<11} | {r['best_distance']:<12}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    # Default to sample PDF or text
    default_doc = "docs/quantum_computing_primer.pdf"
    default_query = "What is the difference between superconducting qubits and trapped ion qubits in terms of gate speed and coherence?"

    doc = sys.argv[1] if len(sys.argv) > 1 else default_doc
    q = sys.argv[2] if len(sys.argv) > 2 else default_query

    run_comparison(doc, q)
