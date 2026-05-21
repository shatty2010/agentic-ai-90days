"""
Day 18 — Basic RAG Pipeline
Week 4: RAG & Semantic Memory | 90-Day Agentic AI Plan

Pipeline: corpus → chunk → embed → ChromaDB → retrieve → augment → Claude
Model:    sentence-transformers (all-MiniLM-L6-v2) for embeddings
          claude-haiku for generation
"""

import os
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./rag_store"
COLLECTION_NAME = "rag_corpus"
CORPUS_FILE = "corpus.txt"
EMBED_MODEL = "all-MiniLM-L6-v2"

st_model = SentenceTransformer(EMBED_MODEL)
claude = anthropic.Anthropic()

SYSTEM_PROMPT = (
    "You are a research assistant.\n"
    "Answer the question using ONLY the context provided below.\n"
    "If the answer is not in the context, say exactly: "
    "'I don't have enough context to answer.'\n"
    "Do not use prior knowledge. Do not guess."
)


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def load_corpus(filepath: str) -> list[str]:
    """Read a plain text file. Returns non-empty paragraphs split on blank lines."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    print(f"Loaded {len(paragraphs)} paragraphs from '{filepath}'")
    return paragraphs


def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
    """
    Fixed-size character chunking. No overlap — that's Day 19.
    Strips empty chunks.
    """
    return [
        text[i : i + chunk_size].strip()
        for i in range(0, len(text), chunk_size)
        if text[i : i + chunk_size].strip()
    ]


def build_rag_store(chunks: list[str]) -> chromadb.Collection:
    """
    Embed chunks and store in a persistent ChromaDB collection.
    Safe to re-run — skips chunks that are already stored.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get()["ids"])
    new_items = [
        (f"chunk_{i}", chunk)
        for i, chunk in enumerate(chunks)
        if f"chunk_{i}" not in existing_ids
    ]

    if new_items:
        new_ids, new_chunks = zip(*new_items)
        new_ids, new_chunks = list(new_ids), list(new_chunks)
        print(f"Embedding {len(new_chunks)} new chunks with {EMBED_MODEL}...")
        vectors = st_model.encode(new_chunks, convert_to_numpy=True).tolist()
        collection.add(ids=new_ids, documents=new_chunks, embeddings=vectors)
        print(f"Stored {len(new_chunks)} chunks in '{COLLECTION_NAME}'.")
    else:
        print(f"Store already populated ({len(existing_ids)} chunks). Skipping insert.")

    return collection


def retrieve(collection: chromadb.Collection, query: str, k: int = 3) -> list[str]:
    """Return the text of the top-k most semantically similar chunks."""
    query_vector = st_model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=k)
    return results["documents"][0]


def generate(query: str, chunks: list[str], print_prompt: bool = False) -> str:
    """
    Build the augmented prompt, optionally print it, call Claude, return the answer.
    Set print_prompt=True on your first run to see exactly what Claude receives.
    """
    context = "\n\n".join(chunks)
    augmented_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    if print_prompt:
        print("\n--- Augmented Prompt ---")
        print(augmented_prompt)
        print("--- End of Prompt ---\n")

    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": augmented_prompt}],
    )
    return response.content[0].text


def run_rag(collection: chromadb.Collection, query: str, k: int = 3, print_prompt: bool = False) -> str:
    """Full pipeline: retrieve → generate → return answer."""
    chunks = retrieve(collection, query, k=k)
    return generate(query, chunks, print_prompt=print_prompt)


# ---------------------------------------------------------------------------
# Experiment: vary k and observe answer quality
# ---------------------------------------------------------------------------

def experiment_k(collection: chromadb.Collection, query: str) -> None:
    """
    Run the same query with k=1, k=3, k=5.
    Observe how answer quality changes with more context.
    k=3 is the production default — understand why after running this.
    """
    print(f"\n{'='*64}")
    print(f"k-experiment — Query: '{query}'")
    print(f"{'='*64}")

    for k in [1, 3, 5]:
        chunks = retrieve(collection, query, k=k)
        answer = generate(query, chunks)
        print(f"\n  k={k}  ({k} chunk{'s' if k > 1 else ''} retrieved)")
        print(f"  Answer: {answer}")
    print()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_tests(collection: chromadb.Collection) -> None:
    tests = [
        # (query, in_corpus)
        ("What is an AI agent and how does it differ from a chatbot?",          True),
        ("How does RAG retrieve relevant information from a knowledge store?",   True),
        ("What is the role of the orchestrator in an agentic system?",           True),
        ("What is the capital city of Australia?",                               False),
        ("How do I prepare a sourdough starter from scratch?",                   False),
    ]

    # Print augmented prompt on the first query only — so you see the pattern once
    first = True

    for query, in_corpus in tests:
        label = "IN-CORPUS " if in_corpus else "OUT-OF-CORPUS"
        print(f"\n{'='*64}")
        print(f"[{label}]  {query}")
        print(f"{'='*64}")

        answer = run_rag(collection, query, k=3, print_prompt=first)
        first = False

        print(f"Answer: {answer}")

        if not in_corpus:
            passed = "I don't have enough context" in answer
            status = "PASS" if passed else "WARN — Claude may have used prior knowledge"
            print(f"Grounding check: {status}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1: load and chunk the corpus
    paragraphs = load_corpus(CORPUS_FILE)
    all_chunks = []
    for para in paragraphs:
        all_chunks.extend(chunk_text(para, chunk_size=300))
    print(f"Total chunks after splitting: {len(all_chunks)}")

    # Step 2: embed and store
    collection = build_rag_store(all_chunks)
    print(f"Chunks in store: {collection.count()}")

    # Step 3: run the 5 test cases
    run_tests(collection)

    # Step 4: k experiment — see how k=1, k=3, k=5 affect quality
    experiment_k(
        collection,
        query="What happens if a tool call fails inside an agent loop?",
    )