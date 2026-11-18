"""
rag_pipeline.py — Stage 2: Interactive RAG with Configurable Chunking Strategies
"""

import os
import glob
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
        raise ValueError("GROQ_API_KEY not found in .env")
    return Groq(api_key=api_key)


def choose_strategy(text: str):
    print("\nSelect Chunking Strategy:")
    print("  1. Fixed Character (blind slice, configurable size/overlap)")
    print("  2. Sentence-based (preserves full sentences)")
    print("  3. Recursive Character (hierarchical: \\n\\n -> \\n -> sentences -> words)")
    print("  4. Token-Aware (LLM token-safe via tiktoken)")

    choice = input("Enter choice [1-4, default=3]: ").strip() or "3"

    if choice == "1":
        size = int(input("Chunk size in chars [default 500]: ").strip() or "500")
        overlap = int(input("Overlap in chars [default 50]: ").strip() or "50")
        return fixed_character_chunker(text, chunk_size=size, chunk_overlap=overlap)

    elif choice == "2":
        sentences = int(input("Sentences per chunk [default 3]: ").strip() or "3")
        overlap = int(input("Sentence overlap [default 1]: ").strip() or "1")
        return sentence_chunker(text, sentences_per_chunk=sentences, sentence_overlap=overlap)

    elif choice == "4":
        tokens = int(input("Max tokens per chunk [default 120]: ").strip() or "120")
        overlap = int(input("Token overlap [default 20]: ").strip() or "20")
        return token_chunker(text, max_tokens=tokens, token_overlap=overlap)

    else:
        size = int(input("Target chunk size in chars [default 500]: ").strip() or "500")
        overlap = int(input("Overlap in chars [default 50]: ").strip() or "50")
        return recursive_character_chunker(text, chunk_size=size, chunk_overlap=overlap)


def main():
    print("=" * 60)
    print(" 🛠️  STAGE 2: INTERACTIVE RAG (CONFIGURABLE CHUNKING)")
    print("=" * 60)

    # 1. Select document
    doc_files = glob.glob("docs/*.txt") + glob.glob("docs/*.pdf")
    if not doc_files:
        print("No documents found in docs/ directory!")
        return

    print("\nAvailable documents:")
    for idx, f in enumerate(doc_files, 1):
        print(f"  [{idx}] {f}")

    doc_choice = input(f"Select document [1-{len(doc_files)}, default=1]: ").strip() or "1"
    selected_doc = doc_files[int(doc_choice) - 1]

    # 2. Load document
    text, meta = load_document(selected_doc)

    # 3. Choose chunker and generate chunks
    chunks = choose_strategy(text)
    print(f"\n[Chunker] Produced {len(chunks)} chunks using strategy '{chunks[0].strategy}'")
    for i, c in enumerate(chunks[:3]):
        print(f"  Chunk {i} ({c.token_count} tokens, {c.char_count} chars): {c.text[:70].strip()}...")

    # 4. Initialize models & Vector store
    embedder = init_embedder()
    groq_client = setup_groq()

    col = create_collection("rag_chunking_col")
    print("\n[Embedder] Generating embeddings...")
    chunk_texts = [c.text for c in chunks]
    embs = get_embeddings(embedder, chunk_texts)
    add_chunks_to_collection(col, chunks, embs, source=meta["source"])
    print("[VectorStore] Chunks indexed in ChromaDB.")

    # 5. Interactive query loop
    print("\n" + "=" * 60)
    print("RAG Ready! Ask questions about your document.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        q = input("Your Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not q:
            continue

        q_emb = get_embeddings(embedder, [q])[0]
        results = retrieve(col, q_emb, top_k=3)

        print(f"\n[Retrieval] Top {len(results)} chunks:")
        for r in results:
            print(f"  • Distance {r['distance']} | Chunk {r['chunk_id']} ({r['token_count']} tok) | {r['text'][:90].strip()}...")

        context = "\n\n---\n\n".join([r["text"] for r in results])
        prompt = (
            f"You are a helpful AI assistant. Answer the user's question based strictly on the provided context.\n"
            f"If the answer cannot be determined from the context, state: 'The context does not contain this information.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {q}\n\nAnswer:"
        )

        res = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        print(f"\n[Answer]:\n{res.choices[0].message.content.strip()}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
