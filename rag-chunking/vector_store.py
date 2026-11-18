"""
vector_store.py — Stage 2: In-Memory Vector Store with Chunk Metadata
"""

import chromadb
from chunkers import Chunk


def create_collection(collection_name: str) -> chromadb.Collection:
    """Create a new ephemeral in-memory ChromaDB collection."""
    client = chromadb.Client()
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_chunks_to_collection(
    collection: chromadb.Collection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    source: str = "doc",
) -> None:
    """Add chunks and embeddings into ChromaDB with rich metadata."""
    ids = [f"{source}_{c.strategy}_{c.chunk_id}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "source": source,
            "chunk_id": c.chunk_id,
            "strategy": c.strategy,
            "char_count": c.char_count,
            "token_count": c.token_count,
        }
        for c in chunks
    ]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )


def retrieve(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:
    """Retrieve top-k closest chunks by cosine distance."""
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
            "chunk_id": meta["chunk_id"],
            "strategy": meta["strategy"],
            "char_count": meta["char_count"],
            "token_count": meta["token_count"],
            "distance": round(dist, 4),
        })

    return retrieved
