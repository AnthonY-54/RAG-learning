"""
retrievers.py — Stage 3: Advanced Retrieval Engine

Implements:
1. Dense Retrieval (Standard Cosine Similarity Top-K)
2. MMR Retrieval (Maximum Marginal Relevance for Diversity)
3. Sparse Retrieval (BM25 for exact keyword/term matching)
4. Hybrid Retrieval (Dense + Sparse fused via Reciprocal Rank Fusion)
5. Cross-Encoder Re-Ranking (FlashRank CPU-optimized Cross-Attention)
"""

import re
import numpy as np
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from chunkers import Chunk


# ============================================================================
# 1. Cosine Similarity & Vector Helpers
# ============================================================================
def cosine_similarity(v1: list[float] | np.ndarray, v2: list[float] | np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors: dot(v1, v2) / (||v1|| * ||v2||)."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================================
# 2. Maximum Marginal Relevance (MMR)
# ============================================================================
def compute_mmr(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    candidate_indices: list[int],
    top_k: int = 3,
    lambda_param: float = 0.5,
) -> list[int]:
    """
    Selects top-k candidates by balancing relevance to query vs diversity
    amongst already selected documents.

    Formula:
        MMR = argmax_{d in Remaining} [ lambda * Sim(d, query) - (1 - lambda) * max_{s in Selected} Sim(d, s) ]

    Args:
        query_embedding: 1D embedding of query
        candidate_embeddings: list of candidate 1D embeddings
        candidate_indices: original chunk IDs
        top_k: number of diverse results to select
        lambda_param: 1.0 = pure relevance (identical to standard top-k),
                      0.0 = maximal diversity,
                      0.5 = balanced (recommended)
    """
    if len(candidate_embeddings) == 0:
        return []

    top_k = min(top_k, len(candidate_embeddings))
    q_vec = np.array(query_embedding, dtype=np.float32)

    # Pre-calculate similarity between query and all candidate chunks
    relevance_scores = [cosine_similarity(q_vec, emb) for emb in candidate_embeddings]

    selected_idxs = []
    remaining_idxs = list(range(len(candidate_embeddings)))

    # First pick: the most relevant candidate to query
    first_choice = int(np.argmax(relevance_scores))
    selected_idxs.append(first_choice)
    remaining_idxs.remove(first_choice)

    # Iteratively select subsequent candidates using MMR formula
    while len(selected_idxs) < top_k and remaining_idxs:
        best_score = -float("inf")
        best_idx = None

        for idx in remaining_idxs:
            relevance = relevance_scores[idx]
            # Max similarity with any already-selected document
            max_redundancy = max(
                cosine_similarity(candidate_embeddings[idx], candidate_embeddings[s])
                for s in selected_idxs
            )
            mmr_score = (lambda_param * relevance) - ((1.0 - lambda_param) * max_redundancy)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_idxs.append(best_idx)
            remaining_idxs.remove(best_idx)

    return [candidate_indices[i] for i in selected_idxs]


# ============================================================================
# 3. BM25 Sparse Keyword Retriever
# ============================================================================
class BM25Retriever:
    """Keyword retriever using Okapi BM25 ranking."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple alphanumeric tokenizer in lowercase."""
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Sort descending by BM25 score
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": float(scores[idx]),
                "rank": rank,
                "strategy": "sparse_bm25",
            })
        return results


# ============================================================================
# 4. Hybrid Retriever (Reciprocal Rank Fusion)
# ============================================================================
class HybridRetriever:
    """
    Fuses Dense Vector Search and BM25 Sparse Search using Reciprocal Rank Fusion (RRF).

    RRF formula:
        Score(doc) = sum_{method} ( 1 / (k + rank_{method}(doc)) )
    """

    def __init__(self, chunks: list[Chunk], chunk_embeddings: list[list[float]]):
        self.chunks = chunks
        self.chunk_embeddings = chunk_embeddings
        self.bm25_retriever = BM25Retriever(chunks)
        self.chunk_map = {c.chunk_id: c for c in chunks}

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 3,
        candidate_pool_size: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        # 1. Sparse BM25 candidate ranks
        sparse_results = self.bm25_retriever.retrieve(query, top_k=candidate_pool_size)
        sparse_ranks = {item["chunk_id"]: item["rank"] for item in sparse_results}

        # 2. Dense Cosine candidate ranks
        dense_scores = [
            (c.chunk_id, cosine_similarity(query_embedding, emb))
            for c, emb in zip(self.chunks, self.chunk_embeddings)
        ]
        # Sort descending by cosine similarity
        dense_scores.sort(key=lambda x: x[1], reverse=True)
        dense_ranks = {
            chunk_id: rank + 1
            for rank, (chunk_id, _) in enumerate(dense_scores[:candidate_pool_size])
        }

        # 3. Fuse scores with Reciprocal Rank Fusion
        all_candidate_ids = set(sparse_ranks.keys()) | set(dense_ranks.keys())
        rrf_scores = {}

        for cid in all_candidate_ids:
            score = 0.0
            if cid in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[cid])
            if cid in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[cid])
            rrf_scores[cid] = score

        # Sort candidate IDs by final RRF score descending
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (cid, score) in enumerate(sorted_candidates, start=1):
            chunk = self.chunk_map[cid]
            results.append({
                "chunk_id": cid,
                "text": chunk.text,
                "score": round(score, 6),
                "rank": rank,
                "strategy": "hybrid_rrf",
                "sparse_rank": sparse_ranks.get(cid, "N/A"),
                "dense_rank": dense_ranks.get(cid, "N/A"),
            })
        return results


# ============================================================================
# 5. Cross-Encoder Re-Ranker (FlashRank)
# ============================================================================
class CrossEncoderReRanker:
    """
    Re-scores candidate passages using a Cross-Encoder (FlashRank ms-marco-TinyBERT).
    Takes a candidate pool (e.g. top 10 from dense or hybrid) and outputs re-ordered top-k.
    """

    def __init__(self):
        print("[ReRanker] Initialising FlashRank Cross-Encoder...")
        self.ranker = Ranker()
        print("[ReRanker] FlashRank ready.")

    def rerank(self, query: str, candidate_chunks: list[dict], top_k: int = 3) -> list[dict]:
        if not candidate_chunks:
            return []

        # Prepare passages format for FlashRank
        passages = [
            {"id": c["chunk_id"], "text": c["text"], "meta": c}
            for c in candidate_chunks
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_passages = self.ranker.rerank(rerank_request)

        results = []
        for rank, p in enumerate(ranked_passages[:top_k], start=1):
            meta = p.get("meta", {})
            results.append({
                "chunk_id": p["id"],
                "text": p["text"],
                "original_rank": meta.get("rank", "N/A"),
                "rerank_score": round(float(p["score"]), 4),
                "rank": rank,
                "strategy": "cross_encoder_rerank",
            })
        return results
