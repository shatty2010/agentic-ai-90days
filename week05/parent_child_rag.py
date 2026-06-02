"""
Day 24 — Parent-Child Chunking
Week 5: Advanced RAG | 90-Day Agentic AI Plan

Two-collection setup:
  child_chunks  — small (100 chars) for precise retrieval
  parent_chunks — large (500 chars) for rich answer context

Retrieve via children, return their parents.
"""

import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_FILE  = "week04/corpus.txt"
CHROMA_PATH  = "./parent_child_store"
EMBED_MODEL  = "all-MiniLM-L6-v2"
PARENT_SIZE  = 500
CHILD_SIZE   = 100

st_model = SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, size: int) -> list[str]:
    return [
        text[i : i + size].strip()
        for i in range(0, len(text), size)
        if text[i : i + size].strip()
    ]


# ---------------------------------------------------------------------------
# Build parent-child store
# ---------------------------------------------------------------------------

def build_parent_child_store(
    corpus_texts: list[str],
) -> tuple[chromadb.Collection, chromadb.Collection]:
    """
    For each corpus paragraph:
      - Create one parent chunk (up to PARENT_SIZE chars)
      - Create N child chunks (CHILD_SIZE each) within that parent

    Parent IDs:  parent_N
    Child IDs:   child_N_M  (N = parent index, M = child index within parent)
    Child metadata: {parent_id: 'parent_N'}
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    for name in ("parent_chunks", "child_chunks"):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    parent_col = client.create_collection(
        "parent_chunks", metadata={"hnsw:space": "cosine"}
    )
    child_col = client.create_collection(
        "child_chunks", metadata={"hnsw:space": "cosine"}
    )

    all_parent_ids, all_parent_texts = [], []
    all_child_ids, all_child_texts, all_child_meta = [], [], []

    for p_idx, para in enumerate(corpus_texts):
        # Build parent chunks from this paragraph
        parent_chunks = chunk_fixed(para, PARENT_SIZE)
        for pc_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"parent_{p_idx}_{pc_idx}"
            all_parent_ids.append(parent_id)
            all_parent_texts.append(parent_text)

            # Build child chunks within this parent
            child_chunks = chunk_fixed(parent_text, CHILD_SIZE)
            for c_idx, child_text in enumerate(child_chunks):
                all_child_ids.append(f"child_{p_idx}_{pc_idx}_{c_idx}")
                all_child_texts.append(child_text)
                all_child_meta.append({"parent_id": parent_id})

    # Store parents
    print(f"Embedding {len(all_parent_texts)} parent chunks...")
    parent_vecs = st_model.encode(all_parent_texts, convert_to_numpy=True).tolist()
    parent_col.add(
        ids=all_parent_ids,
        documents=all_parent_texts,
        embeddings=parent_vecs,
    )

    # Store children
    print(f"Embedding {len(all_child_texts)} child chunks...")
    child_vecs = st_model.encode(all_child_texts, convert_to_numpy=True).tolist()
    child_col.add(
        ids=all_child_ids,
        documents=all_child_texts,
        embeddings=child_vecs,
        metadatas=all_child_meta,
    )

    print(
        f"Store ready — {parent_col.count()} parents, {child_col.count()} children.\n"
    )
    return parent_col, child_col


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_via_children(
    child_col: chromadb.Collection,
    parent_col: chromadb.Collection,
    query: str,
    k: int = 3,
) -> list[dict]:
    """
    Step 1: Find top-k child chunks by similarity.
    Step 2: Look up their parent IDs.
    Step 3: Fetch parent texts, deduplicate.
    Returns [{parent_text, child_text, child_similarity, parent_id}]
    """
    query_vec = st_model.encode(query, convert_to_numpy=True).tolist()
    child_results = child_col.query(query_embeddings=[query_vec], n_results=k * 2)

    seen_parents: set[str] = set()
    hits = []

    for i in range(len(child_results["ids"][0])):
        child_text  = child_results["documents"][0][i]
        similarity  = round(1 - child_results["distances"][0][i], 3)
        parent_id   = child_results["metadatas"][0][i]["parent_id"]

        if parent_id in seen_parents:
            continue  # deduplicate — two children from same parent
        seen_parents.add(parent_id)

        # Fetch parent text
        parent_data = parent_col.get(ids=[parent_id])
        parent_text = parent_data["documents"][0] if parent_data["documents"] else "[parent not found]"

        hits.append({
            "parent_id":        parent_id,
            "parent_text":      parent_text,
            "child_text":       child_text,
            "child_similarity": similarity,
        })

        if len(hits) >= k:
            break

    return hits


def standard_retrieve(
    collection: chromadb.Collection,
    query: str,
    k: int = 3,
) -> list[dict]:
    """Standard 300-char chunk retrieval for comparison."""
    query_vec = st_model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=k)
    return [
        {
            "text":       results["documents"][0][i],
            "similarity": round(1 - results["distances"][0][i], 3),
        }
        for i in range(len(results["ids"][0]))
    ]


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------

def compare(
    query: str,
    pc_results: list[dict],
    std_results: list[dict],
) -> None:
    print(f"\n{'='*68}")
    print(f"Query: '{query}'")
    print(f"{'='*68}")

    print(f"\n  PARENT-CHILD retrieval (child matched → parent returned)")
    print(f"  {'─'*64}")
    for i, r in enumerate(pc_results, 1):
        print(f"  #{i}  child_sim={r['child_similarity']}  [{r['parent_id']}]")
        print(f"       Child:  {r['child_text'][:65].replace(chr(10), ' ')}...")
        print(f"       Parent: {r['parent_text'][:120].replace(chr(10), ' ')}...")
        print()

    print(f"  STANDARD 300-char retrieval")
    print(f"  {'─'*64}")
    for i, r in enumerate(std_results, 1):
        print(f"  #{i}  sim={r['similarity']}  {r['text'][:110].replace(chr(10), ' ')}...")
    print()


# ---------------------------------------------------------------------------
# Standard index (for comparison — 300-char chunks from Day 18 corpus)
# ---------------------------------------------------------------------------

def build_standard_index(corpus_texts: list[str]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection("standard_chunks")
    except Exception:
        pass
    collection = client.create_collection(
        "standard_chunks", metadata={"hnsw:space": "cosine"}
    )
    chunks = []
    for para in corpus_texts:
        chunks.extend(chunk_fixed(para, 300))
    vectors = st_model.encode(chunks, convert_to_numpy=True).tolist()
    collection.add(
        ids=[f"std_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
    )
    return collection


# ---------------------------------------------------------------------------

QUERIES = [
    "Who created Python?",
    "How does safe_call handle tool failures?",
    "What is the difference between short-term and long-term memory in agents?",
    "What does stop_reason end_turn mean?",
]


if __name__ == "__main__":
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        raw = f.read()
    corpus = [p.strip() for p in raw.split("\n\n") if p.strip()]

    parent_col, child_col = build_parent_child_store(corpus)
    standard_col          = build_standard_index(corpus)

    for query in QUERIES:
        pc_results  = retrieve_via_children(child_col, parent_col, query)
        std_results = standard_retrieve(standard_col, query)
        compare(query, pc_results, std_results)
