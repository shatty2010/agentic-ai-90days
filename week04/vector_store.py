"""
Day 17 — Vector Stores with ChromaDB
Week 4: RAG & Semantic Memory | 90-Day Agentic AI Plan

Embeddings: sentence-transformers (local, no API key required)
Model: all-MiniLM-L6-v2 — fast, 384-dim, good general-purpose quality
"""

import os
from datetime import date

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_store"
COLLECTION_NAME = "knowledge_base"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


# 21 documents — AI, Python, AWS, option trading, architecture
DOCUMENTS = [
    {
        "id": "doc_001",
        "text": "Agentic AI systems take actions, use tools, and pursue goals across multiple steps without human intervention at each step.",
        "metadata": {"source": "ai_glossary", "topic": "AI", "date_added": str(date.today())},
    },
    {
        "id": "doc_002",
        "text": "RAG (Retrieval-Augmented Generation) retrieves relevant document chunks and injects them into an LLM prompt before generation.",
        "metadata": {"source": "ai_glossary", "topic": "AI", "date_added": str(date.today())},
    },
    {
        "id": "doc_003",
        "text": "Claude's tool_use API lets you define tools with JSON schemas. Claude decides when to call them and returns structured arguments you execute.",
        "metadata": {"source": "anthropic_docs", "topic": "AI", "date_added": str(date.today())},
    },
    {
        "id": "doc_004",
        "text": "Vector embeddings convert text into lists of numbers that encode semantic meaning. Similar sentences produce vectors that are close in space.",
        "metadata": {"source": "ai_glossary", "topic": "AI", "date_added": str(date.today())},
    },
    {
        "id": "doc_005",
        "text": "Cosine similarity measures the angle between two vectors. A score of 1.0 means identical direction (same meaning), 0 means unrelated.",
        "metadata": {"source": "ai_glossary", "topic": "AI", "date_added": str(date.today())},
    },
    {
        "id": "doc_006",
        "text": "Python was created by Guido van Rossum and first released in 1991. It emphasises code readability — often at the cost of raw performance.",
        "metadata": {"source": "wikipedia", "topic": "Python", "date_added": str(date.today())},
    },
    {
        "id": "doc_007",
        "text": "Python's GIL (Global Interpreter Lock) prevents true parallelism in CPU-bound tasks. Use multiprocessing or asyncio for real concurrency.",
        "metadata": {"source": "python_docs", "topic": "Python", "date_added": str(date.today())},
    },
    {
        "id": "doc_008",
        "text": "Python list comprehensions create lists concisely: [x**2 for x in range(10)]. They are faster than equivalent for-loops and more readable.",
        "metadata": {"source": "python_docs", "topic": "Python", "date_added": str(date.today())},
    },
    {
        "id": "doc_009",
        "text": "AWS Lambda runs code in response to events without provisioning servers. You pay only for execution time, billed in milliseconds.",
        "metadata": {"source": "aws_docs", "topic": "AWS", "date_added": str(date.today())},
    },
    {
        "id": "doc_010",
        "text": "AWS S3 is an object storage service. It stores any file as a flat object with a key. There is no folder hierarchy — prefixes simulate it.",
        "metadata": {"source": "aws_docs", "topic": "AWS", "date_added": str(date.today())},
    },
    {
        "id": "doc_011",
        "text": "AWS API Gateway lets you create and manage REST and WebSocket APIs. It integrates directly with Lambda for a fully serverless HTTP backend.",
        "metadata": {"source": "aws_docs", "topic": "AWS", "date_added": str(date.today())},
    },
    {
        "id": "doc_012",
        "text": "Amazon Bedrock gives access to foundation models from Anthropic, Meta, and others via a single API — no GPU infrastructure required.",
        "metadata": {"source": "aws_docs", "topic": "AWS", "date_added": str(date.today())},
    },
    {
        "id": "doc_013",
        "text": "A call option gives the buyer the right, not the obligation, to buy an asset at a specific strike price before the expiry date.",
        "metadata": {"source": "options_glossary", "topic": "trading", "date_added": str(date.today())},
    },
    {
        "id": "doc_014",
        "text": "A put option gives the buyer the right to sell an asset at the strike price before expiry. Buyers profit when the underlying price falls.",
        "metadata": {"source": "options_glossary", "topic": "trading", "date_added": str(date.today())},
    },
    {
        "id": "doc_015",
        "text": "The Black-Scholes model prices options using five inputs: underlying price, strike price, time to expiry, risk-free rate, and implied volatility.",
        "metadata": {"source": "options_glossary", "topic": "trading", "date_added": str(date.today())},
    },
    {
        "id": "doc_016",
        "text": "Delta measures how much an option's price changes for a $1 move in the underlying. A delta of 0.5 means the option moves $0.50 per $1.",
        "metadata": {"source": "options_glossary", "topic": "trading", "date_added": str(date.today())},
    },
    {
        "id": "doc_017",
        "text": "Theta is the rate at which an option loses value as time passes. Options decay faster in the final 30 days before expiry.",
        "metadata": {"source": "options_glossary", "topic": "trading", "date_added": str(date.today())},
    },
    {
        "id": "doc_018",
        "text": "Microservices structure an application as independently deployable services that communicate over HTTP or message queues.",
        "metadata": {"source": "architecture_glossary", "topic": "architecture", "date_added": str(date.today())},
    },
    {
        "id": "doc_019",
        "text": "The orchestrator pattern uses a central service to coordinate downstream calls in a defined sequence — the same pattern an LLM uses with tool calls.",
        "metadata": {"source": "architecture_glossary", "topic": "architecture", "date_added": str(date.today())},
    },
    {
        "id": "doc_020",
        "text": "ChromaDB is an open-source vector database for storing and querying document embeddings. It supports persistent storage and metadata filtering.",
        "metadata": {"source": "ai_glossary", "topic": "AI", "date_added": str(date.today())},
    },
    {
        "id": "doc_021",
        "text": "sentence-transformers is a Python library for generating dense vector embeddings locally. No API key required — the model runs on your machine.",
        "metadata": {"source": "ai_glossary", "topic": "AI", "date_added": str(date.today())},
    },
]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents. Returns list of float vectors."""
    return model.encode(texts, convert_to_numpy=True).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Returns a float vector."""
    return model.encode(text, convert_to_numpy=True).tolist()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_store(docs: list[dict]) -> chromadb.Collection:
    """
    Create (or reopen) a persistent ChromaDB collection.
    Skips documents already in the store — safe to run multiple times.
    Recreates the collection automatically if the embedding dimension changed.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance → similarity = 1 - distance
    )

    # Detect dimension mismatch (e.g. switched embedding models)
    existing = collection.get()
    if existing["embeddings"] and len(existing["embeddings"][0]) != model.get_embedding_dimension():
        print(
            f"Embedding dimension mismatch "
            f"(stored={len(existing['embeddings'][0])}, "
            f"current={model.get_embedding_dimension()}). "
            f"Recreating collection..."
        )
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    existing_ids = set(collection.get()["ids"])
    new_docs = [d for d in docs if d["id"] not in existing_ids]

    if new_docs:
        print(f"Embedding {len(new_docs)} documents with {MODEL_NAME}...")
        vectors = embed_documents([d["text"] for d in new_docs])
        collection.add(
            ids=[d["id"] for d in new_docs],
            documents=[d["text"] for d in new_docs],
            embeddings=vectors,
            metadatas=[d["metadata"] for d in new_docs],
        )
        print(f"Inserted {len(new_docs)} docs into '{COLLECTION_NAME}'.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already populated ({len(existing_ids)} docs). Skipping insert.")

    return collection


def query_store(
    collection: chromadb.Collection,
    query: str,
    topic_filter: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Return the top-k most semantically similar documents.
    Optionally filter by topic metadata before searching.
    """
    query_vector = embed_query(query)
    where = {"topic": topic_filter} if topic_filter else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return hits


def show_results(query: str, results: list[dict], filter_used: str | None = None) -> None:
    """Pretty-print query results."""
    filter_label = f"  [filter: topic={filter_used}]" if filter_used else ""
    print(f"\nQuery: '{query}'{filter_label}")
    print("-" * 72)
    for rank, hit in enumerate(results, start=1):
        similarity = 1 - hit["distance"]  # cosine space: similarity = 1 - distance
        print(f"  #{rank}  similarity={similarity:.3f}  topic={hit['metadata']['topic']}")
        print(f"        {hit['text'][:90]}...")
        print(f"        source: {hit['metadata']['source']} | id: {hit['id']}")
    print()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_tests(collection: chromadb.Collection) -> None:

    # Tests 1–3: semantic queries, no metadata filter
    print("\n" + "=" * 72)
    print("TESTS 1–3  Semantic queries — no metadata filter")
    print("=" * 72)

    semantic_queries = [
        "How do AI agents decide which tools to call?",
        "What happens to an option's value as the expiry date approaches?",
        "How does AWS run code without provisioning a server?",
    ]
    for q in semantic_queries:
        results = query_store(collection, q)
        show_results(q, results)

    # Tests 4–5: queries with topic metadata filter
    print("=" * 72)
    print("TESTS 4–5  Queries with topic metadata filter")
    print("=" * 72)

    filtered_queries = [
        ("How does Python handle parallel execution and the GIL?", "Python"),
        ("What is the difference between a call and a put option?", "trading"),
    ]
    for q, topic in filtered_queries:
        results = query_store(collection, q, topic_filter=topic)
        show_results(q, results, filter_used=topic)

    # Test 6: delete a doc, verify it no longer appears
    print("=" * 72)
    print("TEST 6  Delete doc_020, confirm it no longer appears in results")
    print("=" * 72)

    target_id = "doc_020"
    before = query_store(collection, "What is ChromaDB used for?")
    top_before = before[0]
    print(f"Before deletion — top result: {top_before['id']}")
    print(f"  Text: {top_before['text'][:70]}...")

    collection.delete(ids=[target_id])
    print(f"\nDeleted: {target_id}")

    after = query_store(collection, "What is ChromaDB used for?")
    returned_ids = [r["id"] for r in after]

    assert target_id not in returned_ids, f"ERROR: {target_id} still appears after deletion"
    print(f"After deletion — top result: {after[0]['id']}")
    print(f"  Text: {after[0]['text'][:70]}...")
    print("\nDeletion confirmed.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    collection = build_store(DOCUMENTS)
    print(f"Total docs in store: {collection.count()}")
    #run_tests(collection)

    print("\n" + "=" * 72)
    print("COLLECTION INFO")
    print("=" * 72)
    print(f"  Name      : {collection.name}")
    print(f"  ID        : {collection.id}")
    print(f"  Doc count : {collection.count()}")
    print(f"  Metadata  : {collection.metadata}")

    print("\n" + "=" * 72)
    # Fetch everything in one single database hit
    results = collection.get()

    # Loop through the zip of the parallel lists
    for doc_id, text, metadata in zip(results['ids'], results['documents'], results['metadatas']):
        print(f"  ID: {doc_id}")
        print(f"  Text: {text[:70]}...")
        print(f"  Metadata: {metadata}")
        print("-" * 72)
