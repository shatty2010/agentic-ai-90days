"""
Day 22 — Re-ranking with Cross-Encoders
Week 5: Advanced RAG | 90-Day Agentic AI Plan

Two-stage retrieval:
  Stage 1 — bi-encoder  : ChromaDB nearest-neighbour, top-20 candidates (fast)
  Stage 2 — cross-encoder: re-rank candidates by relevance score (accurate)
"""

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

CORPUS_FILE   = "week04/corpus.txt"
CHROMA_PATH   = "./rerank_store"
COLLECTION    = "rerank_corpus"
BI_ENCODER    = "all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

bi_model    = SentenceTransformer(BI_ENCODER)
cross_model = CrossEncoder(CROSS_ENCODER)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def load_corpus(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    return [p.strip() for p in raw.split("\n\n") if p.strip()]


def build_index(corpus_texts: list[str]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Embedding {len(corpus_texts)} documents with {BI_ENCODER}...")
    vectors = bi_model.encode(corpus_texts, convert_to_numpy=True).tolist()
    collection.add(
        ids=[f"doc_{i}" for i in range(len(corpus_texts))],
        documents=corpus_texts,
        embeddings=vectors,
    )
    print(f"Index ready — {collection.count()} docs.\n")
    return collection


# ---------------------------------------------------------------------------
# Two-stage retrieval
# ---------------------------------------------------------------------------

def retrieve_candidates(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 20,
) -> list[dict]:
    """Stage 1: fast bi-encoder retrieval. Returns top-k candidates."""
    query_vec = bi_model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    return [
        {
            "text":        results["documents"][0][i],
            "bi_rank":     i + 1,
            "bi_similarity": round(1 - results["distances"][0][i], 3),
        }
        for i in range(len(results["ids"][0]))
    ]


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """
    Stage 2: cross-encoder scores each (query, chunk) pair.
    Returns top_n candidates re-sorted by cross-encoder score.
    """
    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_model.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for new_rank, (score, candidate) in enumerate(ranked[:top_n], start=1):
        results.append({
            "text":           candidate["text"],
            "cross_rank":     new_rank,
            "cross_score":    round(float(score), 4),
            "bi_rank":        candidate["bi_rank"],
            "bi_similarity":  candidate["bi_similarity"],
            "rank_change":    candidate["bi_rank"] - new_rank,  # positive = moved up
        })
    return results


def run_pipeline(
    collection: chromadb.Collection,
    query: str,
    candidate_k: int = 20,
    top_n: int = 5,
) -> tuple[list[dict], list[dict]]:
    """
    Full two-stage pipeline.
    Returns (bi_encoder_top5, reranked_top5) for comparison.
    """
    candidates = retrieve_candidates(collection, query, top_k=candidate_k)
    reranked   = rerank(query, candidates, top_n=top_n)
    return candidates[:top_n], reranked


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(
    query: str,
    before: list[dict],
    after: list[dict],
) -> None:
    print(f"\nQuery: '{query}'")
    print(f"{'─'*72}")
    print(f"  {'Stage':<14}  {'Rank':<5}  {'Score':<8}  Text (first 60 chars)")
    print(f"  {'─'*14}  {'─'*5}  {'─'*8}  {'─'*60}")

    for i in range(max(len(before), len(after))):
        # Before (bi-encoder)
        if i < len(before):
            b = before[i]
            print(
                f"  {'Bi-encoder':<14}  #{b['bi_rank']:<4}  {b['bi_similarity']:<8}  "
                f"{b['text'][:60].replace(chr(10), ' ')}..."
            )
        # After (cross-encoder)
        if i < len(after):
            a = after[i]
            arrow = f"↑{a['rank_change']}" if a['rank_change'] > 0 else (
                    f"↓{abs(a['rank_change'])}" if a['rank_change'] < 0 else "  =")
            print(
                f"  {'Cross-encoder':<14}  #{a['cross_rank']:<4}  {a['cross_score']:<8}  "
                f"{a['text'][:60].replace(chr(10), ' ')}...  [{arrow}]"
            )
        if i < max(len(before), len(after)) - 1:
            print()
    print()


# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

QUERIES = [
    "What is the stopping condition for an agent loop?",
    "Why is fine-tuning worse than RAG for frequently updated data?",
    "What does Claude return when it wants to call a tool?",
    "How does memory work across multiple agent sessions?",
]


if __name__ == "__main__":
    corpus     = load_corpus(CORPUS_FILE)
    collection = build_index(corpus)

    print(f"{'='*72}")
    print("TWO-STAGE RETRIEVAL: Bi-encoder → Cross-encoder Re-ranking")
    print(f"{'='*72}")

    for query in QUERIES:
        before, after = run_pipeline(collection, query, candidate_k=20, top_n=5)
        print_comparison(query, before, after)
