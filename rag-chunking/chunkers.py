"""
chunkers.py — Stage 2: Pluggable Chunking Strategies

This module implements 4 fundamental chunking strategies:

1. Fixed-Size Character Chunking:
   - Slices strings purely by character count.
   - Fast, but blind to sentence and word boundaries (often cuts words in half).

2. Sentence-based Chunking:
   - Splits on natural punctuation (. ! ?).
   - Groups a target number of complete sentences per chunk with sentence-level overlap.
   - Guarantees grammatical completeness, but chunk length varies with sentence length.

3. Recursive Character Chunking:
   - The gold standard in production RAG (LangChain / LlamaIndex style).
   - Tries to split on largest semantic units first (\n\n paragraphs),
     falling back to lines (\n), then sentences (. ), then words (" "),
     and finally characters only if needed.
   - Preserves semantic structure while respecting maximum chunk size.

4. Token-Aware Chunking:
   - Measures chunk size in actual LLM tokens using tiktoken (cl100k_base).
   - Prevents accidental token-context overflow when passing context to embedding models or LLMs.
"""

import re
from dataclasses import dataclass
import tiktoken

# Initialize a standard BPE tokenizer (used by modern LLMs)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    text: str
    chunk_id: int
    strategy: str
    char_count: int
    token_count: int


# ============================================================================
# Strategy 1: Fixed-Size Character Chunking (Blind Slice)
# ============================================================================
def fixed_character_chunker(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Chunk]:
    """Slices text strictly by character length."""
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_str = text[start:end]
        tokens = len(_TOKENIZER.encode(chunk_str))

        chunks.append(
            Chunk(
                text=chunk_str,
                chunk_id=chunk_id,
                strategy="fixed_character",
                char_count=len(chunk_str),
                token_count=tokens,
            )
        )
        chunk_id += 1
        start += chunk_size - chunk_overlap

    return chunks


# ============================================================================
# Strategy 2: Sentence-Based Chunking
# ============================================================================
def sentence_chunker(
    text: str, sentences_per_chunk: int = 3, sentence_overlap: int = 1
) -> list[Chunk]:
    """
    Splits text by sentence boundary regex, grouping N sentences per chunk.
    Never splits in the middle of a sentence.
    """
    if sentences_per_chunk <= sentence_overlap:
        raise ValueError("sentences_per_chunk must be greater than sentence_overlap")

    # Regular expression to split on sentence boundaries while keeping punctuation
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(sentences):
        end = start + sentences_per_chunk
        group = sentences[start:end]
        chunk_str = " ".join(group)
        tokens = len(_TOKENIZER.encode(chunk_str))

        chunks.append(
            Chunk(
                text=chunk_str,
                chunk_id=chunk_id,
                strategy="sentence",
                char_count=len(chunk_str),
                token_count=tokens,
            )
        )
        chunk_id += 1
        start += sentences_per_chunk - sentence_overlap

    return chunks


# ============================================================================
# Strategy 3: Recursive Character Chunking (Hierarchical)
# ============================================================================
def recursive_character_chunker(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] = None,
) -> list[Chunk]:
    """
    Splits text recursively by trying separators in order:
    1. "\n\n" (paragraphs)
    2. "\n"   (lines)
    3. ". "   (sentences)
    4. " "    (words)
    5. ""     (characters fallback)
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(t: str, seps: list[str]) -> list[str]:
        final_pieces = []
        # Find the first separator present in text
        chosen_sep = ""
        new_seps = []
        for i, s in enumerate(seps):
            if s == "" or s in t:
                chosen_sep = s
                new_seps = seps[i + 1:]
                break

        splits = t.split(chosen_sep) if chosen_sep else list(t)

        good_splits = []
        for s in splits:
            if chosen_sep and chosen_sep != " ":
                s = s + chosen_sep
            if len(s) < chunk_size:
                good_splits.append(s)
            else:
                if new_seps:
                    good_splits.extend(_split_text(s, new_seps))
                else:
                    good_splits.append(s)

        # Merge splits up to chunk_size with overlap
        merged = []
        current = ""
        for piece in good_splits:
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current.strip():
                    merged.append(current.strip())
                # Start new chunk with overlap from the end of current
                overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                current = overlap_text + piece

        if current.strip():
            merged.append(current.strip())

        return merged

    raw_chunks = _split_text(text, separators)
    chunks = []
    for i, c in enumerate(raw_chunks):
        chunks.append(
            Chunk(
                text=c,
                chunk_id=i,
                strategy="recursive",
                char_count=len(c),
                token_count=len(_TOKENIZER.encode(c)),
            )
        )
    return chunks


# ============================================================================
# Strategy 4: Token-Aware Chunking (LLM Context Safe)
# ============================================================================
def token_chunker(
    text: str, max_tokens: int = 120, token_overlap: int = 20
) -> list[Chunk]:
    """
    Splits text strictly based on LLM token counts using tiktoken.
    Guarantees no chunk exceeds max_tokens.
    """
    if max_tokens <= token_overlap:
        raise ValueError("max_tokens must be greater than token_overlap")

    tokens = _TOKENIZER.encode(text)
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk_token_slice = tokens[start:end]
        chunk_str = _TOKENIZER.decode(chunk_token_slice)

        chunks.append(
            Chunk(
                text=chunk_str,
                chunk_id=chunk_id,
                strategy="token_aware",
                char_count=len(chunk_str),
                token_count=len(chunk_token_slice),
            )
        )
        chunk_id += 1
        start += max_tokens - token_overlap

    return chunks
