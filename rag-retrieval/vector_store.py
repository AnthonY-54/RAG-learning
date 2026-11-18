"""
vector_store.py — Stage 3: Vector Store with Dense & MMR Retrieval
"""

import chromadb
from chunkers import Chunk
from retrievers import compute_mmr


def create_collection(collection_name: str) -> chromadb.Collection:
    """Create a new ephemeral ChromaDB collection."""
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
    ids = [f"{source}_{c.chunk_id}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {"source": source, "chunk_id": c.chunk_id, "char_count": c.char_count}
        for c in chunks
    ]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )


def retrieve_dense(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:
    """Standard top-k dense retrieval by cosine distance."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        retrieved.append({
            "chunk_id": meta["chunk_id"],
            "text": doc,
            "source": meta["source"],
            "distance": round(dist, 4),
            "rank": rank,
            "strategy": "dense_cosine",
        })
    return retrieved


def retrieve_mmr(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 3,
    candidate_fetch_k: int = 15,
    lambda_param: float = 0.5,
) -> list[dict]:
    """
    Diverse retrieval using Maximum Marginal Relevance (MMR).
    Fetches candidate_fetch_k candidate embeddings first, then applies MMR.
    """
    # 1. Fetch a wider pool of candidate chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_fetch_k,
        include=["documents", "metadatas", "embeddings", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    embs = results["embeddings"][0]
    dists = results["distances"][0]

    candidate_ids = [m["chunk_id"] for m in metas]

    # 2. Compute MMR indices
    selected_chunk_ids = compute_mmr(
        query_embedding=query_embedding,
        candidate_embeddings=embs,
        candidate_indices=candidate_ids,
        top_k=top_k,
        lambda_param=lambda_param,
    )

    # 3. Assemble results in MMR order
    chunk_dict = {
        m["chunk_id"]: {"text": d, "source": m["source"], "distance": round(dist, 4)}
        for d, m, dist in zip(docs, metas, dists)
    }

    retrieved = []
    for rank, cid in enumerate(selected_chunk_ids, start=1):
        info = chunk_dict[cid]
        retrieved.append({
            "chunk_id": cid,
            "text": info["text"],
            "source": info["source"],
            "distance": info["distance"],
            "rank": rank,
            "strategy": f"dense_mmr(lambda={lambda_param})",
        })
    return retrieved
