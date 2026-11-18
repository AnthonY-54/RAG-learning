"""
embedder.py — Stage 1: Generating Embeddings Locally

Responsibility: Convert text strings into embedding vectors using FastEmbed.

What is an embedding?
  A list of floating-point numbers (e.g., 384 floats) that represents
  the *semantic meaning* of the input text. Two texts with similar meanings
  will have vectors that are close together in high-dimensional space.

Why FastEmbed?
  - Runs 100% locally on your machine via ONNX Runtime (CPU-optimized, fast).
  - Zero API keys, zero network latency, zero rate limits.
  - Model: "BAAI/bge-small-en-v1.5" (384 dimensions, one of the best compact retrieval models).
"""

from fastembed import TextEmbedding


def init_embedder(model_name: str = "BAAI/bge-small-en-v1.5") -> TextEmbedding:
    """
    Initialise the local embedding model.
    The weights (~67MB) are downloaded on first run and cached locally.
    """
    print(f"[Embedder] Loading local model '{model_name}'...")
    embedder = TextEmbedding(model_name=model_name)
    print("[Embedder] Model ready.")
    return embedder


def get_embeddings(embedder: TextEmbedding, texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of strings.

    Args:
        embedder: The initialised TextEmbedding instance.
        texts:    List of text chunks or query strings.

    Returns:
        List of embedding vectors (each vector is a list of floats).
    """
    # embedder.embed() yields numpy arrays; convert each to a Python list for ChromaDB
    embeddings = [e.tolist() for e in embedder.embed(texts)]
    print(f"[Embedder] Generated embeddings for {len(embeddings)} text item(s)")
    return embeddings
