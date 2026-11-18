"""
embedder.py — Stage 2: FastEmbed Local Embeddings
"""

from fastembed import TextEmbedding


def init_embedder(model_name: str = "BAAI/bge-small-en-v1.5") -> TextEmbedding:
    """Initialise local ONNX FastEmbed model."""
    print(f"[Embedder] Initialising model '{model_name}'...")
    embedder = TextEmbedding(model_name=model_name)
    print("[Embedder] Ready.")
    return embedder


def get_embeddings(embedder: TextEmbedding, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of strings."""
    embeddings = [e.tolist() for e in embedder.embed(texts)]
    return embeddings
