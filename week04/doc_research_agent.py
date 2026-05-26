"""
Day 20 — Capstone: Document Research Agent
Week 4: RAG & Semantic Memory | 90-Day Agentic AI Plan

Two retrieval paths in one agent loop:
  Path 1 — search_docs  : ChromaDB RAG over local corpus
  Path 2 — live tools   : get_weather, convert_currency, calculate
"""

import os
import time
import json
import math
import requests
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CORPUS_FILE  = "week04/corpus.txt"
CHROMA_PATH  = "./capstone_store"
COLLECTION   = "doc_corpus"
EMBED_MODEL  = "all-MiniLM-L6-v2"

st_model = SentenceTransformer(EMBED_MODEL)
claude   = anthropic.Anthropic()

# Built once in main(), referenced by search_docs
doc_collection: chromadb.Collection | None = None


# ---------------------------------------------------------------------------
# RAG — doc store
# ---------------------------------------------------------------------------

def _chunk_recursive(
    text: str,
    size: int = 300,
    overlap: int = 50,
    min_size: int = 50,
) -> list[str]:
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
                if len(piece) > size and rest:
                    chunks.extend(_split(piece, rest))
                    current = ""
                else:
                    current = piece
        if current:
            chunks.append(current)
        return [c for c in chunks if len(c) >= min_size]

    raw = _split(text, ["\n\n", ". ", " "])
    result = []
    for i, chunk in enumerate(raw):
        if i > 0 and overlap > 0:
            tail = raw[i - 1][-overlap:]
            chunk = (tail + " " + chunk).strip()
        result.append(chunk)
    return result


def build_doc_store(corpus_path: str) -> chromadb.Collection:
    """Load corpus, chunk with recursive strategy, embed, store in ChromaDB."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    chunks = _chunk_recursive(corpus, size=300, overlap=50)
    print(f"Corpus: {len(chunks)} chunks from '{corpus_path}'")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    vectors = st_model.encode(chunks, convert_to_numpy=True).tolist()
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
    )
    print(f"Doc store ready — {collection.count()} chunks stored.\n")
    return collection


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def search_docs(query: str) -> dict:
    """
    RAG retrieval path: embed query → ChromaDB → return top-3 chunks.
    The agent uses this for questions about the local document corpus.
    """
    query_vector = st_model.encode(query, convert_to_numpy=True).tolist()
    results = doc_collection.query(query_embeddings=[query_vector], n_results=3)
    chunks  = results["documents"][0]
    scores  = [round(1 - d, 3) for d in results["distances"][0]]
    return {"chunks": chunks, "similarity_scores": scores}


def get_weather(city: str) -> dict:
    """Live path: current weather via Open-Meteo (geocoding + forecast)."""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=5,
    ).json()
    if not geo.get("results"):
        return {"error": f"City not found: {city}"}
    loc = geo["results"][0]
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "current_weather": True,
        },
        timeout=5,
    ).json()
    cw = weather["current_weather"]
    return {
        "city": city,
        "temp_celsius": cw["temperature"],
        "wind_kph": cw["windspeed"],
    }


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Live path: currency conversion via Frankfurter (ECB rates, free, no key)."""
    resp = requests.get(
        "https://api.frankfurter.app/latest",
        params={"amount": amount, "from": from_currency.upper(), "to": to_currency.upper()},
        timeout=5,
    ).json()
    if "rates" not in resp:
        return {"error": resp.get("message", "Conversion failed")}
    result = resp["rates"][to_currency.upper()]
    return {
        "amount": amount,
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "result": result,
        "rate": round(result / amount, 6),
    }


def calculate(expression: str) -> dict:
    """Local: evaluate a mathematical expression in a sandboxed namespace."""
    safe_globals = {"__builtins__": {}, "math": math}
    safe_chars   = set("0123456789+-*/(). eE")
    if not all(c in safe_chars for c in expression):
        return {"error": f"Unsafe characters in expression: '{expression}'"}
    try:
        result = eval(expression, safe_globals)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "search_docs",
        "description": (
            "Search the local document corpus for factual information. "
            "Always call this tool first for any factual or knowledge question "
            "before relying on your own training knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'London'."},
            },
            "required": ["city"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount between two currencies using live ECB rates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount":        {"type": "number",  "description": "Amount to convert."},
                "from_currency": {"type": "string",  "description": "Source currency code, e.g. 'USD'."},
                "to_currency":   {"type": "string",  "description": "Target currency code, e.g. 'EUR'."},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a safe mathematical expression, e.g. '2**16' or '3847 * 0.15'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A Python math expression."},
            },
            "required": ["expression"],
        },
    },
]

TOOL_REGISTRY = {
    "search_docs":       search_docs,
    "get_weather":       get_weather,
    "convert_currency":  convert_currency,
    "calculate":         calculate,
}

SYSTEM_PROMPT = """You are a research assistant with two retrieval paths.

For questions about real-time data (weather, currency, calculations):
    use get_weather, convert_currency, or calculate.

For ALL factual or knowledge questions — regardless of topic — always call
search_docs first to check the local corpus before relying on your own knowledge.

If a question needs both live data and document context, call both tools.
Always state which tool(s) you used and summarise what they returned."""


# ---------------------------------------------------------------------------
# Agent infrastructure
# ---------------------------------------------------------------------------

tool_call_log: list[dict] = []


def safe_call(tool_name: str, fn, **kwargs) -> tuple[dict, float]:
    """
    Execute a tool function. Catches all exceptions.
    Returns (result_dict, elapsed_seconds).
    """
    start = time.time()
    try:
        result = fn(**kwargs)
    except requests.Timeout:
        result = {"error": f"{tool_name} timed out — retry or use a different query"}
    except Exception as exc:
        result = {"error": f"{tool_name} failed: {type(exc).__name__}: {exc}"}
    elapsed = round(time.time() - start, 3)
    tool_call_log.append({"tool": tool_name, "elapsed_s": elapsed, "ok": "error" not in result})
    return result, elapsed


def dispatch_tools(blocks: list) -> list[dict]:
    """
    Execute all tool_use blocks from a Claude response.
    Handles single and parallel calls in one pass.
    Returns a list of tool_result messages ready to inject.
    """
    tool_results = []
    for block in blocks:
        if block.type != "tool_use":
            continue
        fn = TOOL_REGISTRY.get(block.name)
        if not fn:
            result = {"error": f"Unknown tool: {block.name}"}
            elapsed = 0.0
        else:
            result, elapsed = safe_call(block.name, fn, **block.input)

        print(f"  [{block.name}]  args={block.input}  →  {result}  ({elapsed}s)")
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(result),
        })
    return tool_results


def run_agent(question: str, max_steps: int = 10) -> str:
    """
    Main agent loop.
    Sends question to Claude → dispatches tools → injects results → loops.
    Returns final answer when stop_reason == 'end_turn'.
    """
    messages = [{"role": "user", "content": question}]
    tool_call_log.clear()

    for step in range(1, max_steps + 1):
        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract text from the final response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "(no text in final response)"

        if response.stop_reason == "tool_use":
            print(f"\n  Step {step} — tool calls:")
            tool_results = dispatch_tools(response.content)
            messages.append({"role": "user", "content": tool_results})
        else:
            return f"Unexpected stop_reason: {response.stop_reason}"

    return "Max steps reached without final answer."


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TESTS = [
   (
        "According to the documents, what is the role of the orchestrator in an agentic system?",
        "search_docs",
    ),
    (
        "What is the current weather in Tokyo?",
        "get_weather",
    ),
    (
        "How much is 250 USD in EUR?",
        "convert_currency",
    ),
    (
        "What is 2 to the power of 16?",
        "calculate",
    ),
    (
        "What is the book 'The Great Gatsby' about, and what is the current temperature in London?",
        "search_docs + get_weather",
    ),
]


def run_tests() -> None:
    for i, (question, expected_path) in enumerate(TESTS, start=1):
        print(f"\n{'='*64}")
        print(f"Test {i} — Expected path: {expected_path}")
        print(f"Q: {question}")
        print(f"{'='*64}")

        answer = run_agent(question)

        print(f"\nAnswer:\n{answer}")
        print(f"\nTool call log: {tool_call_log}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build doc store once — uses corpus.txt from Day 18
    doc_collection = build_doc_store(CORPUS_FILE)

    # Run the 5 capstone test cases
    run_tests()