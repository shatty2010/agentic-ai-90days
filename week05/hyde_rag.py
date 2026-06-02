"""
Day 23 — HyDE: Hypothetical Document Embeddings
Week 5: Advanced RAG | 90-Day Agentic AI Plan

Instead of embedding the user query directly, generate a hypothetical
answer with Claude, embed that, and retrieve against the hypothesis embedding.
Compare standard vs HyDE retrieval on the same queries.
"""

import os
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CORPUS_FILE = "week04/corpus.txt"
CHROMA_PATH = "./hyde_store"
COLLECTION  = "hyde_corpus"
EMBED_MODEL = "all-MiniLM-L6-v2"

st_model = SentenceTransformer(EMBED_MODEL)
claude   = anthropic.Anthropic()


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
    vectors = st_model.encode(corpus_texts, convert_to_numpy=True).tolist()
    collection.add(
        ids=[f"doc_{i}" for i in range(len(corpus_texts))],
        documents=corpus_texts,
        embeddings=vectors,
    )
    print(f"Index ready — {collection.count()} docs.\n")
    return collection


# ---------------------------------------------------------------------------
# HyDE core
# ---------------------------------------------------------------------------

def generate_hypothesis(query: str) -> str:
    """
    Ask Claude to write a short hypothetical answer to the query.
    The hypothesis is used for retrieval only — not shown to the user.
    It doesn't need to be correct, just plausible and in the right domain.
    """
    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Write a short, factual paragraph (2-3 sentences) that would answer "
                f"this question. Be specific and use domain vocabulary.\n\n"
                f"Question: {query}"
            ),
        }],
    )
    return response.content[0].text.strip()


def _retrieve(collection: chromadb.Collection, text: str, k: int = 3) -> list[dict]:
    """Core retrieval — embed text, return top-k chunks with scores."""
    vec = st_model.encode(text, convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=[vec], n_results=k)
    return [
        {
            "text":       results["documents"][0][i],
            "similarity": round(1 - results["distances"][0][i], 3),
        }
        for i in range(len(results["ids"][0]))
    ]


def standard_retrieve(collection: chromadb.Collection, query: str, k: int = 3) -> list[dict]:
    """Embed the raw query and retrieve."""
    return _retrieve(collection, query, k=k)


def hyde_retrieve(
    collection: chromadb.Collection,
    query: str,
    k: int = 3,
) -> tuple[list[dict], str]:
    """
    Generate a hypothesis, embed it, retrieve.
    Returns (results, hypothesis_text) — hypothesis printed for transparency.
    """
    hypothesis = generate_hypothesis(query)
    results    = _retrieve(collection, hypothesis, k=k)
    return results, hypothesis


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(c["text"] for c in chunks)
    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=(
            "Answer using ONLY the context below. "
            "If the answer is not in the context, say: "
            "'I don't have enough context to answer.'"
        ),
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        }],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_retrieval(collection: chromadb.Collection, query: str) -> None:
    print(f"\n{'='*68}")
    print(f"Query: '{query}'")
    print(f"{'='*68}")

    # Standard retrieval
    std_results = standard_retrieve(collection, query)
    print(f"\n  STANDARD (raw query embedding)")
    print(f"  {'─'*64}")
    for i, r in enumerate(std_results, 1):
        print(f"  #{i}  sim={r['similarity']}  {r['text'][:70].replace(chr(10), ' ')}...")

    # HyDE retrieval
    hyde_results, hypothesis = hyde_retrieve(collection, query)
    print(f"\n  HyDE (hypothesis embedding)")
    print(f"  Hypothesis: {hypothesis[:120].replace(chr(10), ' ')}...")
    print(f"  {'─'*64}")
    for i, r in enumerate(hyde_results, 1):
        print(f"  #{i}  sim={r['similarity']}  {r['text'][:70].replace(chr(10), ' ')}...")

    # Final answers
    std_answer  = generate_answer(query, std_results)
    hyde_answer = generate_answer(query, hyde_results)
    print(f"\n  Standard answer: {std_answer[:150]}...")
    print(f"  HyDE answer:     {hyde_answer[:150]}...")


# ---------------------------------------------------------------------------
# Test queries — short/vague ones where HyDE should help most
# ---------------------------------------------------------------------------

QUERIES = [
    "memory in agents",
    "options theta",
    "chunking trade-offs",
    "What is the role of safe_call?",  # longer query — HyDE may not help here
]


if __name__ == "__main__":
    corpus     = load_corpus(CORPUS_FILE)
    collection = build_index(corpus)

    for query in QUERIES:
        compare_retrieval(collection, query)
