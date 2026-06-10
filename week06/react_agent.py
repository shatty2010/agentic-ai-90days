"""
Day 26 — ReAct Pattern: Reasoning and Acting
Week 6: Agent Architectures | 90-Day Agentic AI Plan

Extends the Week 3 tool_use loop with explicit Thought/Action/Observation traces.
Every reasoning step is captured and printed — making the agent debuggable.
"""

import os
import json
import time
import math
import requests
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

claude    = anthropic.Anthropic()
st_model  = SentenceTransformer("all-MiniLM-L6-v2")

# RAG store from Week 4 capstone (rebuild if needed)
CHROMA_PATH = "./capstone_store"
COLLECTION  = "doc_corpus"

try:
    _client     = chromadb.PersistentClient(path=CHROMA_PATH)
    doc_col     = _client.get_collection(COLLECTION)
except Exception:
    doc_col = None


# ---------------------------------------------------------------------------
# Tools (same as Week 4 doc_research_agent)
# ---------------------------------------------------------------------------

def search_docs(query: str) -> dict:
    if doc_col is None:
        return {"error": "Doc store not built. Run week04/doc_research_agent.py first."}
    vec     = st_model.encode(query, convert_to_numpy=True).tolist()
    results = doc_col.query(query_embeddings=[vec], n_results=3)
    chunks  = results["documents"][0]
    scores  = [round(1 - d, 3) for d in results["distances"][0]]
    return {"chunks": chunks, "similarity_scores": scores}


def get_weather(city: str) -> dict:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}, timeout=5,
    ).json()
    if not geo.get("results"):
        return {"error": f"City not found: {city}"}
    loc     = geo["results"][0]
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": loc["latitude"], "longitude": loc["longitude"], "current_weather": True},
        timeout=5,
    ).json()
    cw = weather["current_weather"]
    return {"city": city, "temp_celsius": cw["temperature"], "wind_kph": cw["windspeed"]}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    resp = requests.get(
        "https://api.frankfurter.app/latest",
        params={"amount": amount, "from": from_currency.upper(), "to": to_currency.upper()},
        timeout=5,
    ).json()
    if "rates" not in resp:
        return {"error": resp.get("message", "Conversion failed")}
    result = resp["rates"][to_currency.upper()]
    return {"amount": amount, "from": from_currency.upper(), "to": to_currency.upper(),
            "result": result, "rate": round(result / amount, 6)}


def calculate(expression: str) -> dict:
    safe_globals = {"__builtins__": {}, "math": math}
    safe_chars   = set("0123456789+-*/(). eE")
    if not all(c in safe_chars for c in expression):
        return {"error": f"Unsafe expression: '{expression}'"}
    try:
        return {"expression": expression, "result": eval(expression, safe_globals)}  # noqa: S307
    except Exception as exc:
        return {"error": str(exc)}


TOOL_REGISTRY = {
    "search_docs":      search_docs,
    "get_weather":      get_weather,
    "convert_currency": convert_currency,
    "calculate":        calculate,
}

TOOL_SCHEMAS = [
    {
        "name": "search_docs",
        "description": "Search the local document corpus about Agentic AI, Python, AWS, option trading, and architecture.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount between two currencies using live ECB rates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount":        {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency":   {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a safe mathematical expression, e.g. '2**20' or '100 * 1.15'.",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
]

SYSTEM_PROMPT = """You are a research assistant that thinks step by step.

Before every tool call, write your reasoning starting with "Thought:" on its own line.
After receiving a tool result, summarise what you learned starting with "Observation:" on its own line.
When you have enough information, write "Final Answer:" followed by your complete response.

Be explicit. Show your reasoning. Every Thought should explain WHY you're calling that tool."""


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

def extract_thoughts(response) -> list[str]:
    """Pull all text blocks from a response — these are the reasoning traces."""
    thoughts = []
    for block in response.content:
        if hasattr(block, "text") and block.text.strip():
            thoughts.append(block.text.strip())
    return thoughts


def safe_call(name: str, fn, **kwargs) -> tuple[dict, float]:
    start = time.time()
    try:
        result = fn(**kwargs)
    except requests.Timeout:
        result = {"error": f"{name} timed out"}
    except Exception as exc:
        result = {"error": f"{name}: {type(exc).__name__}: {exc}"}
    return result, round(time.time() - start, 3)


def dispatch_tools(blocks) -> list[dict]:
    tool_results = []
    for block in blocks:
        if block.type != "tool_use":
            continue
        fn     = TOOL_REGISTRY.get(block.name)
        result, elapsed = safe_call(block.name, fn, **block.input) if fn else ({"error": f"Unknown tool: {block.name}"}, 0)

        # Print Observation
        print(f"\n  ── Observation ({block.name}, {elapsed}s) ──")
        print(f"  {json.dumps(result, indent=2)}")

        tool_results.append({
            "type":        "tool_result",
            "tool_use_id": block.id,
            "content":     json.dumps(result),
        })
    return tool_results


def run_react_agent(question: str, max_steps: int = 10) -> str:
    messages = [{"role": "user", "content": question}]
    print(f"\n{'='*64}")
    print(f"Question: {question}")
    print(f"{'='*64}")

    for step in range(1, max_steps + 1):
        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Print reasoning traces
        thoughts = extract_thoughts(response)
        for thought in thoughts:
            print(f"\n  ── Step {step} · Thought ──")
            print(f"  {thought}")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        if response.stop_reason == "tool_use":
            # Print action
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n  ── Action ──")
                    print(f"  {block.name}({json.dumps(block.input)})")

            tool_results = dispatch_tools(response.content)
            messages.append({"role": "user", "content": tool_results})

    return "Max steps reached."


# ---------------------------------------------------------------------------
# Test questions — multi-step to make reasoning traces rich
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What is the weather in Paris, and how much is 200 EUR in USD?",
    "According to the documents, what is stop_reason end_turn, and what is 2**20?",
    "Find the current temperature in London and calculate how many degrees below boiling point that is.",
]

if __name__ == "__main__":
    for q in QUESTIONS:
        answer = run_react_agent(q)
        print(f"\n  ── Final Answer ──")
        print(f"  {answer}")
        print()
