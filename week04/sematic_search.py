"""
Day 16 — Embeddings & Semantic Search
---------------------------------------
Goal: Understand embeddings and cosine similarity by building semantic search
from scratch — no vector database, no framework. Just sentence-transformers.

What this script does:
  1. Embeds a 20-item FAQ corpus locally using sentence-transformers (all-MiniLM-L6-v2)
  2. Implements cosine similarity manually with pure Python math
  3. Searches the corpus by meaning, not keywords
  4. Returns top-k matches with scores for any query

Key concept: two texts with similar *meanings* produce vectors that are
numerically close. Cosine similarity measures that closeness — 1.0 is identical,
0 is unrelated. This is semantic search.

.NET analogy: embeddings are like a hash function that preserves semantic distance
instead of identity. Two similar inputs → nearby vectors. The vector space has
geometry, and that geometry encodes meaning.

Run:
    pip uninstall scikit-learn scipy -y   # remove DLL-heavy deps not needed here
    pip install sentence-transformers
    python sematic_search.py
"""

import math
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

FAQ_CORPUS = [
    # AI / GenAI
    "What is an LLM? A large language model is a neural network trained on massive text data to predict and generate human-like text.",
    "What is prompt engineering? It is the practice of crafting inputs to LLMs to reliably get high-quality outputs.",
    "What is RAG? Retrieval-Augmented Generation retrieves relevant documents and injects them into a prompt before generation.",
    "What is an AI agent? An agent is an LLM in a loop that uses tools to take actions and complete multi-step tasks.",
    "What is the difference between fine-tuning and RAG? Fine-tuning updates model weights; RAG retrieves external knowledge at inference time without changing the model.",

    # Python
    "What is a Python decorator? A decorator wraps a function to add behaviour before or after it runs, using the @syntax.",
    "What is the GIL in Python? The Global Interpreter Lock allows only one thread to execute Python bytecode at a time, limiting CPU parallelism.",
    "What is the difference between a list and a tuple in Python? Lists are mutable; tuples are immutable and slightly faster for read-only data.",
    "What is async/await in Python? It enables non-blocking I/O operations — your code can pause and resume without blocking the thread.",
    "What is type hinting in Python? Type hints annotate variable and function types for static analysis tools and developer clarity.",

    # AWS
    "What is AWS Lambda? Lambda is a serverless compute service — you run code in response to events without managing servers.",
    "What is the difference between SQS and SNS? SQS is a message queue for point-to-point async communication; SNS is a pub/sub service for fan-out delivery.",
    "What is AWS API Gateway? A managed service that creates, publishes, and secures HTTP APIs in front of Lambda functions or other backends.",
    "What is IAM in AWS? Identity and Access Management controls who can do what in your AWS account — users, roles, and policies.",
    "What is AWS Step Functions? A serverless orchestration service that chains Lambda functions and services into stateful workflows.",

    # Option trading
    "What is a call option? A call gives the buyer the right to purchase an asset at a fixed price before expiry.",
    "What is implied volatility? Implied volatility is the market's forecast of how much an asset's price will move, derived from option prices.",
    "What is theta decay in options? Theta is the rate at which an option loses value as time passes, all else being equal.",
    "What is a covered call strategy? Selling a call option on a stock you own, collecting premium while capping your upside.",

    # Career
    "What skills are most valued for AI engineering roles abroad? Agentic AI, RAG pipelines, tool-use patterns, and the ability to explain systems without abstraction layers.",
]


def embed_corpus(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts locally — no API, no rate limits."""
    print(f"  Embedding {len(texts)} corpus items...")
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string locally."""
    return model.encode([query], normalize_embeddings=True)[0].tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity = dot(a, b) / (||a|| × ||b||)

    Returns a float in [-1, 1].
      1.0  → identical direction (same meaning)
      0.0  → orthogonal (unrelated)
     -1.0  → opposite direction (rare for natural language)

    Vectors from all-MiniLM-L6-v2 are already normalised, so this reduces
    to dot(a, b) — but we compute it explicitly to make the maths visible.
    Pure Python math: no numpy, no DLL dependencies.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    query: str,
    corpus_texts: list[str],
    corpus_vectors: list[list[float]],
    top_k: int = 3
) -> list[dict]:
    """
    Embed the query, compute cosine similarity against every corpus vector,
    return the top_k results sorted by score descending.

    Returns a list of dicts: {rank, score, text}
    """
    q_vec = embed_query(query)
    scores = [
        {"rank": 0, "score": cosine_similarity(q_vec, v), "text": t}
        for v, t in zip(corpus_vectors, corpus_texts)
    ]
    scores.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(scores[:top_k], 1):
        item["rank"] = i
    return scores[:top_k]


def print_results(query: str, results: list[dict]) -> None:
    print(f"\n  Query: \"{query}\"")
    print(f"  {'─'*60}")
    for r in results:
        short_text = r["text"][:90] + "..." if len(r["text"]) > 90 else r["text"]
        print(f"  #{r['rank']}  score={r['score']:.4f}  {short_text}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*64)
    print("  Day 16 — Semantic Search over FAQ Corpus")
    print("="*64)

    # Build corpus vectors once — in a real system you'd cache these to disk
    corpus_vectors = embed_corpus(FAQ_CORPUS)
    print(f"  Vector dimensions: {len(corpus_vectors[0])}")

    # ── Test queries ──────────────────────────────────────────────────────────
    # Design: queries that don't share exact words with corpus items.
    # If semantic search works, the right items still rank first.

    test_queries = [
        # 1. No exact word overlap with correct answer — pure semantic match
        "How does the government of France relate to Paris?",

        # 2. Uses 'agents' but the right answer uses 'agent' — tests stemming via semantics
        "What are AI agents and how do they work?",

        # 3. Option trading — doesn't use the word 'theta' or 'decay'
        "Why does holding an option for a long time cost me money even if the price doesn't move?",

        # 4. AWS — uses informal language, not technical terms
        "I want to run code without spinning up a server on AWS — what should I use?",

        # 5. Deliberately off-topic — nothing in the corpus should match well
        "What is the best recipe for chocolate cake?",
    ]

    for q in test_queries:
        results = search(q, FAQ_CORPUS, corpus_vectors, top_k=3)
        print_results(q, results)

    # ── Similarity comparison — build intuition ───────────────────────────────
    print(f"\n\n{'='*64}")
    print("  Similarity comparison: same meaning, different words")
    print(f"{'='*64}")

    pairs = [
        ("Paris is the capital of France",
         "France's capital city is Paris",
         "Same meaning, different word order — expect ~0.97+"),

        ("What is a call option?",
         "A call gives the buyer the right to purchase an asset at a fixed price before expiry.",
         "Question vs answer for same topic — expect ~0.90+"),

        ("AWS Lambda is a serverless compute service",
         "Python is a programming language",
         "Unrelated topics — expect ~0.60 or lower"),

        ("What is RAG?",
         "What is fine-tuning?",
         "Related AI concepts — expect moderate similarity ~0.80"),
    ]

    all_texts = [p[0] for p in pairs] + [p[1] for p in pairs]
    vecs = model.encode(all_texts, normalize_embeddings=True).tolist()
    n = len(pairs)

    print()
    for i, (a_text, b_text, note) in enumerate(pairs):
        score = cosine_similarity(vecs[i], vecs[n + i])
        print(f"  [{note}]")
        print(f"  A: {a_text[:60]}")
        print(f"  B: {b_text[:60]}")
        print(f"  Similarity: {score:.4f}\n")
