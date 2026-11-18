"""
vector_store.py — Stage 1: Storing and Retrieving Vectors

Responsibility: Store chunk text + embeddings in ChromaDB, and retrieve
the most semantically similar chunks for a given query.

What is ChromaDB?
  A lightweight, open-source vector database that runs entirely in-process
  (no server needed for local dev). It stores:
    - The text of each chunk (the "document")
    - The embedding vector for that chunk
    - Optional metadata (e.g., source filename, chunk index)

How retrieval works:
  You embed the query → pass the query vector to Chroma →
  Chroma computes cosine similarity between the query vector and all stored
  chunk vectors → returns the top-k most similar chunks.

We will NOT let ChromaDB compute embeddings for us (it can, via embedding
functions). We do it manually so you see the full pipeline.
"""

import chromadb


def create_collection(collection_name: str = "rag_basic") -> chromadb.Collection:
    """
    Create an in-memory ChromaDB client and collection.

    In-memory means data is lost when the program exits.
    (We'll add persistence in Stage 4.)

    Args:
        collection_name: Name of the collection (like a table name).

    Returns:
        A ChromaDB Collection object.
    """
    client = chromadb.Client()  # ephemeral, in-memory
    collection = client.create_collection(
        name=collection_name,
        # We use cosine similarity (default is L2 distance).
        # Cosine similarity is standard for semantic search.
        metadata={"hnsw:space": "cosine"}
    )
    print(f"[VectorStore] Created collection '{collection_name}'")
    return collection


def add_chunks(
    collection: chromadb.Collection,
    chunks: list[str],
    embeddings: list[list[float]],
    source: str = "document",
) -> None:
    """
    Store chunks and their embeddings in the ChromaDB collection.

    Args:
        collection:  The ChromaDB collection to store into.
        chunks:      List of text chunks.
        embeddings:  Corresponding embedding vectors.
        source:      Source identifier (e.g., filename) stored as metadata.
    """
    # ChromaDB requires a unique string ID per document.
    ids = [f"{source}_chunk_{i}" for i in range(len(chunks))]

    # Metadata allows filtering later (e.g., "only search chunks from doc X")
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"[VectorStore] Stored {len(chunks)} chunks from '{source}'")


def retrieve(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks for a given query embedding.

    Args:
        collection:      The ChromaDB collection to search.
        query_embedding: The embedding vector of the user's query.
        top_k:           Number of results to return.

    Returns:
        List of dicts, each with 'text', 'source', 'chunk_index', 'distance'.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": round(dist, 4),  # lower = more similar (cosine distance)
        })

    return retrieved
