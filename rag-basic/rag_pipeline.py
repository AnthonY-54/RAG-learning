"""
rag_pipeline.py — Stage 1: The Full RAG Pipeline (Groq + Local Embeddings)

Responsibility: Wire together loader → chunker → embedder → vector store → Groq LLM.

Pipeline Flow:
  1. Ingestion:
     docs/sample.txt ──► load ──► chunk ──► embed (FastEmbed local) ──► store in ChromaDB
  2. Query:
     User question ──► embed question (FastEmbed) ──► retrieve top-k chunks ──► build prompt
     ──► Groq API (llama-3.3-70b-versatile) ──► return answer
"""

import os
from dotenv import load_dotenv
from groq import Groq

from loader import load_text_file
from chunker import chunk_text
from embedder import init_embedder, get_embeddings
from vector_store import create_collection, add_chunks, retrieve


def setup_groq() -> Groq:
    """Load GROQ_API_KEY from .env and return an initialised Groq client."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not found or not set in .env! "
            "Please create a .env file with: GROQ_API_KEY=gsk_..."
        )
    client = Groq(api_key=api_key)
    print("[Setup] Groq client initialised")
    return client


def ingest(embedder, filepath: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Ingest a document: load → chunk → embed → store.
    """
    print("\n--- INGESTION PHASE ---")

    # Step 1: Load text
    text = load_text_file(filepath)

    # Step 2: Split text into overlapping chunks
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 3: Embed chunks locally using FastEmbed
    print("[Embedder] Generating local embeddings for chunks...")
    embeddings = get_embeddings(embedder, chunks)

    # Step 4: Store chunks & vectors into ChromaDB
    collection = create_collection()
    source_name = os.path.basename(filepath)
    add_chunks(collection, chunks, embeddings, source=source_name)

    return collection


def query(groq_client: Groq, embedder, collection, question: str, top_k: int = 3) -> str:
    """
    Query the RAG system:
    1. Embed query vector
    2. Cosine similarity search in ChromaDB
    3. Construct prompt with retrieved chunks as context
    4. Call Groq (llama-3.3-70b-versatile) to synthesize answer
    """
    print("\n--- RETRIEVAL PHASE ---")

    # Step 5: Embed query using the same local embedding model
    print(f"[Query] Embedding question: '{question}'")
    query_embedding = get_embeddings(embedder, [question])[0]

    # Step 6: Retrieve top-k similar chunks
    retrieved_chunks = retrieve(collection, query_embedding, top_k=top_k)

    print(f"[Retrieval] Top {top_k} chunks retrieved:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  [{i+1}] distance={chunk['distance']} | "
              f"chunk_index={chunk['chunk_index']} | "
              f"{chunk['text'][:80].strip()}...")

    print(f"\n--- GENERATION PHASE (Groq {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}) ---")

    # Step 7: Build the stuffed prompt
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, Chunk {c['chunk_index']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    system_instruction = (
        "You are a helpful assistant. Answer the user's question based ONLY on the "
        "provided context. If the answer is not contained in the context, strictly respond: "
        "\"I don't have enough information in the provided document to answer that.\""
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    # Step 8: Call Groq for fast inference
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        model=model_name,
        temperature=0.1,  # Low temperature = more grounded, less hallucination
    )

    answer = chat_completion.choices[0].message.content
    return answer


def main():
    # --- Configuration ---
    DOCUMENT_PATH = "docs/sample.txt"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K = 3

    # --- Setup ---
    groq_client = setup_groq()
    embedder = init_embedder()

    # --- Ingest ---
    collection = ingest(embedder, DOCUMENT_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # --- Interactive Q&A loop ---
    print("\n" + "="*50)
    print("RAG Basic (Groq + Local FastEmbed) — Ready!")
    print("Type your questions below. Type 'quit' to exit.")
    print("="*50 + "\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        answer = query(groq_client, embedder, collection, question, top_k=TOP_K)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
