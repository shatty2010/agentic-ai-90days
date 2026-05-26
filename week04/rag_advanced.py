"""
Day 19 — Advanced RAG: Chunking & Retrieval Quality
Week 4: RAG & Semantic Memory | 90-Day Agentic AI Plan

Three chunking strategies on the same corpus, stored in separate ChromaDB
collections, same 3 queries run against all three, side-by-side comparison.
"""

import re
import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_FILE = "week04/corpus.txt"
CHROMA_PATH = "./rag_advanced_store"
EMBED_MODEL  = "all-MiniLM-L6-v2"

st_model = SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, size: int = 300) -> list[str]:
    """
    Split every N characters. Fast, simple, semantically blind.
    Cuts mid-sentence wherever the counter hits size.
    """
    return [
        text[i : i + size].strip()
        for i in range(0, len(text), size)
        if text[i : i + size].strip()
    ]


def chunk_sentences(text: str, min_size: int = 50) -> list[str]:
    """
    Split on sentence-ending punctuation (., !, ?).
    Discards chunks shorter than min_size — they produce weak embeddings.

    Known limitation: false splits on abbreviations like 'e.g.', 'Dr.', '1.5.'
    Use a proper tokeniser (spaCy / NLTK) in production.
    """
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if len(s.strip()) >= min_size]


def chunk_recursive(
    text: str,
    size: int = 300,
    overlap: int = 50,
    min_size: int = 50,
) -> list[str]:
    """
    Split on the largest separator first (\n\n, then '. ', then ' ').
    Respects document structure — paragraphs stay whole if they fit.
    Falls back to finer splits only when needed.
    Appends last [overlap] chars of chunk N to the start of chunk N+1.
    """
    def _split(text: str, separators: list[str]) -> list[str]:
        sep, *rest = separators
        pieces = text.split(sep)
        chunks, current = [], ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            candidate = (current + " " + piece).strip() if current else piece
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Piece itself is too long — recurse into next separator
                if len(piece) > size and rest:
                    chunks.extend(_split(piece, rest))
                    current = ""
                else:
                    current = piece

        if current:
            chunks.append(current)

        return [c for c in chunks if len(c) >= min_size]

    raw = _split(text, ["\n\n", ". ", " "])

    # Apply overlap: prepend last [overlap] chars of previous chunk
    result = []
    for i, chunk in enumerate(raw):
        if i > 0 and overlap > 0:
            tail = raw[i - 1][-overlap:]
            chunk = (tail + " " + chunk).strip()
        result.append(chunk)

    return result


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def build_collection(name: str, chunks: list[str]) -> chromadb.Collection:
    """Embed chunks and store in a named ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop and recreate so re-runs always reflect the current chunks
    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"  Embedding {len(chunks)} chunks for '{name}'...")
    vectors = st_model.encode(chunks, convert_to_numpy=True).tolist()
    collection.add(
        ids=[f"{name}_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
    )
    return collection


def retrieve_with_scores(
    collection: chromadb.Collection,
    query: str,
    k: int = 5,
) -> list[dict]:
    """Return top-k results with similarity scores and chunk index."""
    query_vector = st_model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=k)

    hits = []
    for i in range(len(results["ids"][0])):
        raw_id = results["ids"][0][i]
        chunk_index = int(raw_id.split("_")[-1])
        hits.append({
            "chunk_index": chunk_index,
            "text": results["documents"][0][i],
            "similarity": round(1 - results["distances"][0][i], 3),
        })
    return hits


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def chunk_stats(name: str, chunks: list[str]) -> None:
    lengths = [len(c) for c in chunks]
    print(
        f"  {name:<12}  count={len(chunks):>3}  "
        f"min={min(lengths):>4}  max={max(lengths):>4}  "
        f"avg={sum(lengths)//len(lengths):>4}  chars"
    )


def print_comparison(
    query: str,
    results_by_strategy: dict[str, list[dict]],
) -> None:
    """Print a side-by-side retrieval table for a single query."""
    print(f"\nQuery: '{query}'")
    print("-" * 72)
    print(f"  {'Strategy':<12}  {'Rank':<5}  {'Similarity':<12}  Text (first 65 chars)")
    print(f"  {'-'*12}  {'-'*5}  {'-'*12}  {'-'*65}")

    for strategy, hits in results_by_strategy.items():
        for rank, hit in enumerate(hits[:3], start=1):  # show top-3 per strategy
            prefix = f"  {strategy:<12}" if rank == 1 else f"  {'':<12}"
            print(
                f"{prefix}  #{rank:<4}  {hit['similarity']:<12}  "
                f"{hit['text'][:65].replace(chr(10), ' ')}..."
            )
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Step 1: load corpus
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = f.read()
    print(f"Corpus loaded — {len(corpus)} characters\n")

    # Step 2: apply three chunking strategies
    fixed_chunks     = chunk_fixed(corpus, size=300)
    sentence_chunks  = chunk_sentences(corpus)
    recursive_chunks = chunk_recursive(corpus, size=300, overlap=50)

    print("Chunk statistics:")
    chunk_stats("fixed",     fixed_chunks)
    chunk_stats("sentence",  sentence_chunks)
    chunk_stats("recursive", recursive_chunks)
    print()

    # Step 3: build a separate ChromaDB collection per strategy
    print("Building collections:")
    fixed_col     = build_collection("fixed",     fixed_chunks)
    sentence_col  = build_collection("sentence",  sentence_chunks)
    recursive_col = build_collection("recursive", recursive_chunks)
    print()

    # Step 4: same 3 queries against all 3 strategies
    queries = [
        "How does an AI agent decide which tool to call next?",
        "What is the role of the orchestrator in an agentic loop?",
        "Why does chunk overlap improve retrieval quality?",
    ]

    collections = {
        "fixed":     fixed_col,
        "sentence":  sentence_col,
        "recursive": recursive_col,
    }

    print("=" * 72)
    print("RETRIEVAL COMPARISON")
    print("=" * 72)

    for query in queries:
        results_by_strategy = {
            name: retrieve_with_scores(col, query, k=5)
            for name, col in collections.items()
        }
        print_comparison(query, results_by_strategy)

    # ---------------------------------------------------------------------------
    # WINNER ANALYSIS — fill this in after running and reading the output
    # ---------------------------------------------------------------------------
    # Strategy that won:    recursive
    # Why it won:           Paragraph-level splits keep semantic units intact;
    #                       overlap prevents boundary answers from being missed.
    # When fixed wins:      Very short, uniform corpora where sentence length
    #                       is consistent and boundaries rarely matter.
    # ---------------------------------------------------------------------------