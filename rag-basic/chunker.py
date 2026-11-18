"""
chunker.py — Stage 1: Text Chunking

Responsibility: Split a large text into smaller, overlapping pieces (chunks).

Why chunk?
  Embedding models have token limits (e.g., 2048 tokens).
  More importantly: smaller chunks = more precise retrieval.
  If you embed an entire 10-page doc as one vector, a question about page 3
  will match the whole doc vaguely — not the specific paragraph on page 3.

Why overlap?
  If a key sentence falls exactly at a chunk boundary, it would be split.
  Overlapping chunks ensure boundary content appears in at least one full chunk.

Parameters:
  chunk_size   : number of CHARACTERS per chunk (not tokens — simpler for now)
  chunk_overlap: how many characters the next chunk shares with the previous one
"""


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text:         The full document text.
        chunk_size:   Max characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        A list of text chunks (strings).
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap) so the next chunk
        # re-reads the last `chunk_overlap` characters of this chunk.
        start += chunk_size - chunk_overlap

    print(f"[Chunker] Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks
