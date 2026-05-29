"""
Day 21 — Hybrid Search: BM25 + Vector + RRF
Week 5: Advanced RAG | 90-Day Agentic AI Plan

Combines keyword (BM25) and semantic (vector) retrieval.
Merges ranked results using Reciprocal Rank Fusion.
"""

import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CORPUS_FILE = "week04/corpus.txt"
CHROMA_PATH = "./hybrid_store"
COLLECTION  = "hybrid_corpus"
EMBED_MODEL = "all-MiniLM-L6-v2"

st_model = SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    return [p.strip() for p in raw.split("\n\n") if p.strip()]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def tokenise(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(corpus_texts: list[str]) -> BM25Okapi:
    tokenised = [tokenise(doc) for doc in corpus_texts]
    return BM25Okapi(tokenised)


def search_bm25(
    bm25: BM25Okapi,
    corpus_texts: list[str],
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Return top-k BM25 results with rank and score. Zero-score docs are excluded."""
    scores = bm25.get_scores(tokenise(query))
    ranked = [
        (idx, s)
        for idx, s in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        if s > 0
    ][:top_k]
    return [
        {"text": corpus_texts[idx], "bm25_rank": rank + 1, "bm25_score": round(score, 4)}
        for rank, (idx, score) in enumerate(ranked)
    ]


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def build_vector_index(corpus_texts: list[str]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Embedding {len(corpus_texts)} docs for vector index...")
    vectors = st_model.encode(corpus_texts, convert_to_numpy=True).tolist()
    collection.add(
        ids=[f"doc_{i}" for i in range(len(corpus_texts))],
        documents=corpus_texts,
        embeddings=vectors,
    )
    return collection


def search_vector(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Return top-k vector results with rank and similarity."""
    query_vec = st_model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    return [
        {
            "text": results["documents"][0][i],
            "vector_rank": i + 1,
            "similarity": round(1 - results["distances"][0][i], 3),
        }
        for i in range(len(results["ids"][0]))
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge two ranked lists using RRF.
    score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vector(d))
    Documents absent from a list get rank = len(list) + 1 (last place).
    """
    # Absent-from-list penalty: rank just beyond the actual list length,
    # but never lower than the candidate pool size so empty lists don't
    # accidentally assign rank-1 to every document.
    candidate_k = max(len(bm25_results), len(vector_results), 1)
    bm25_default_rank   = max(len(bm25_results),   candidate_k) + 1
    vector_default_rank = max(len(vector_results), candidate_k) + 1

    # Build text → rank maps
    bm25_rank_map   = {r["text"]: r["bm25_rank"]   for r in bm25_results}
    vector_rank_map = {r["text"]: r["vector_rank"]  for r in vector_results}

    # Collect all unique documents
    all_texts = set(bm25_rank_map) | set(vector_rank_map)

    fused = []
    for text in all_texts:
        rb = bm25_rank_map.get(text, bm25_default_rank)
        rv = vector_rank_map.get(text, vector_default_rank)
        rrf_score = 1 / (k + rb) + 1 / (k + rv)
        fused.append({
            "text":       text,
            "rrf_score":  round(rrf_score, 6),
            "bm25_rank":  rb if rb != bm25_default_rank  else "-",
            "vector_rank": rv if rv != vector_default_rank else "-",
        })

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def hybrid_search(
    bm25: BM25Okapi,
    collection: chromadb.Collection,
    corpus_texts: list[str],
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Full hybrid search pipeline. Returns RRF-ranked top-k results."""
    bm25_results   = search_bm25(bm25, corpus_texts, query, top_k=10)
    vector_results = search_vector(collection, query, top_k=10)
    fused          = reciprocal_rank_fusion(bm25_results, vector_results)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(
    query: str,
    bm25_res: list[dict],
    vector_res: list[dict],
    hybrid_res: list[dict],
) -> None:
    print(f"\nQuery: '{query}'")
    print(f"{'─'*72}")
    print(f"  {'Method':<9}  {'Rank':<5}  {'Score':<8}  Text (first 65 chars)")
    print(f"  {'─'*9}  {'─'*5}  {'─'*8}  {'─'*65}")

    rows = [
        ("BM25",   [(r["bm25_rank"],   r["bm25_score"],  r["text"]) for r in bm25_res[:3]]),
        ("Vector", [(r["vector_rank"], r["similarity"],  r["text"]) for r in vector_res[:3]]),
        ("Hybrid", [(i+1,              r["rrf_score"],   r["text"]) for i, r in enumerate(hybrid_res[:3])]),
    ]

    for method, results in rows:
        for i, (rank, score, text) in enumerate(results):
            label = f"  {method:<9}" if i == 0 else f"  {'':<9}"
            print(f"{label}  #{rank:<4}  {score:<8}  {text[:65].replace(chr(10), ' ')}...")
        print()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

QUERIES = [
    ("BM25 algorithm",                        "BM25 should rank exact match highest"),
    ("how do agents decide which tool to use", "Vector should handle semantic query"),
    ("Python GIL concurrency",                "Mixed — both should contribute"),
    ("memory",                                "Vague — compare all three approaches"),
]


if __name__ == "__main__":
    corpus = load_corpus(CORPUS_FILE)
    print(f"Corpus: {len(corpus)} paragraphs\n")

    bm25       = build_bm25(corpus)
    collection = build_vector_index(corpus)

    print(f"\n{'='*72}")
    print("HYBRID SEARCH — BM25 + VECTOR + RRF")
    print(f"{'='*72}")

    for query, note in QUERIES:
        bm25_res   = search_bm25(bm25, corpus, query, top_k=10)
        vector_res = search_vector(collection, query, top_k=10)
        hybrid_res = hybrid_search(bm25, collection, corpus, query, top_k=5)
        print_comparison(query, bm25_res, vector_res, hybrid_res)
        print(f"  Note: {note}\n")
