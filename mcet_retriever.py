"""
MCET RAG Hybrid Retriever
==========================

Ties together:
  - MCET_synthetic_qa_pairs.json   (fast-path: pre-written Q -> A pairs)
  - MCET_chunks.json                (fallback: raw text chunks for vector search)

Strategy
--------
1. FAST PATH: embed the incoming user query and compare it against the
   embeddings of all pre-generated synthetic questions. If the best match
   clears FAST_PATH_THRESHOLD, return that pre-written answer immediately
   (cheap, fast, consistent wording for common questions).

2. FALLBACK: if no synthetic question is a close enough match, fall back to
   standard semantic search over the raw chunk store and return the best
   matching chunk(s) as retrieved context (for an LLM to then compose an
   answer from).

Backends
--------
- Default: sentence-transformers, all-MiniLM-L6-v2 (local, free, no API key).
- Optional: OpenAI embeddings (set EMBED_BACKEND="openai" + OPENAI_API_KEY env var).

Usage
-----
    python mcet_retriever.py "Who is the HOD of CSE?"
    python mcet_retriever.py "What is the fee for lateral entry IT?"

Or import and use programmatically:
    from mcet_retriever import HybridRetriever
    r = HybridRetriever()
    result = r.retrieve("Who is the chairman of MCET?")
"""

import os
import sys
import json
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QA_PAIRS_PATH = os.path.join(os.path.dirname(__file__), "MCET_synthetic_qa_pairs.json")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "MCET_chunks.json")

EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "sentence-transformers")  # or "openai"
FAST_PATH_THRESHOLD = 0.72   # cosine similarity cutoff to trust the fast-path answer
FALLBACK_TOP_K = 3           # number of chunks to return when falling back


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------
class SentenceTransformerBackend:
    """Local, free embedding backend. Downloads the model once, then runs offline."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class OpenAIBackend:
    """Optional backend using OpenAI's embedding API. Requires OPENAI_API_KEY."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # normalize for cosine similarity via dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)


def get_backend():
    if EMBED_BACKEND == "openai":
        return OpenAIBackend()
    return SentenceTransformerBackend()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_qa_pairs(path: str = QA_PAIRS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["qa_pairs"]


def load_chunks(path: str = CHUNKS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self):
        self.backend = get_backend()

        self.qa_pairs = load_qa_pairs()
        self.chunks = load_chunks()

        # Pre-embed fast-path questions
        self._qa_questions = [qa["question"] for qa in self.qa_pairs]
        self._qa_embeddings = self.backend.embed(self._qa_questions)

        # Pre-embed fallback chunks
        self._chunk_ids = list(self.chunks.keys())
        self._chunk_texts = list(self.chunks.values())
        self._chunk_embeddings = self.backend.embed(self._chunk_texts)

        # Consistency cache for queries
        self._cache = {}

    @staticmethod
    def _cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        # vectors are already normalized, so cosine similarity == dot product
        return matrix @ query_vec

    def retrieve(self, query: str, force_refresh: bool = True) -> dict:
        query_hash = hash(query.lower().strip())
        
        # Check cache if force_refresh is False
        if not force_refresh and query_hash in self._cache:
            return self._cache[query_hash]

        query_vec = self.backend.embed([query])[0]

        # --- Fast path ---
        qa_sims = self._cosine_sim(query_vec, self._qa_embeddings)
        best_idx = int(np.argmax(qa_sims))
        best_score = float(qa_sims[best_idx])

        if best_score >= FAST_PATH_THRESHOLD:
            matched = self.qa_pairs[best_idx]
            result = {
                "path": "fast_path",
                "matched_question": matched["question"],
                "answer": matched["answer"],
                "chunk_id": matched["chunk_id"],
                "confidence": matched.get("confidence", "unspecified"),
                "similarity": round(best_score, 4),
            }
            self._cache[query_hash] = result
            return result

        # --- Fallback: vector search over raw chunks ---
        chunk_sims = self._cosine_sim(query_vec, self._chunk_embeddings)
        top_idx = np.argsort(chunk_sims)[::-1][:FALLBACK_TOP_K]

        results = [
            {
                "chunk_id": self._chunk_ids[i],
                "text": self._chunk_texts[i],
                "similarity": round(float(chunk_sims[i]), 4),
            }
            for i in top_idx
        ]

        result = {
            "path": "fallback_vector_search",
            "best_fast_path_score": round(best_score, 4),  # shown for debugging/tuning threshold
            "retrieved_chunks": results,
        }
        self._cache[query_hash] = result
        return result


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def _pretty_print(result: dict):
    if result["path"] == "fast_path":
        print(f"\n[FAST PATH] matched: \"{result['matched_question']}\" (sim={result['similarity']})")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']} | source chunk: {result['chunk_id']}")
    else:
        print(f"\n[FALLBACK] no fast-path match (best score={result['best_fast_path_score']}) - "
              f"top {len(result['retrieved_chunks'])} chunks retrieved:")
        for c in result["retrieved_chunks"]:
            print(f"  - ({c['similarity']}) [{c['chunk_id']}] {c['text']}")
        print("\n-> Pass these chunks as context to your LLM to compose the final answer.")


if __name__ == "__main__":
    retriever = HybridRetriever()

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        # demo queries: one designed to hit fast-path, one designed to hit fallback
        queries = [
            "Who is the HOD of CSE?",
            "What documents do I need to submit for lateral entry admission?",
        ]

    for q in queries:
        print("=" * 70)
        print(f"Query: {q}")
        result = retriever.retrieve(q)
        _pretty_print(result)
