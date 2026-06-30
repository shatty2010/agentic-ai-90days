"""
Day 35 — Capstone: Multi-Step Research Agent with HITL
Week 7: LangGraph | 90-Day Agentic AI Plan

Graph:
  START → planner → researcher → analyst → [INTERRUPT] → reporter → END
                                                            ↓ if rejected
                                                           END (with note)

HITL: interrupt_after=["analyst"] — human reviews analysis before report is written.
Structured output: ResearchReport via submit_report tool.
"""

import os
import json
import math as _math
import anthropic
import chromadb
import requests
from typing import TypedDict, Literal, Annotated
import operator
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

claude    = anthropic.Anthropic()
MODEL     = "claude-haiku-4-5-20251001"
st_model  = SentenceTransformer("all-MiniLM-L6-v2")

# Optional doc store — works if week04 corpus was built
try:
    _client = chromadb.PersistentClient(path="./week04_store")
    doc_col = _client.get_collection("doc_corpus")
except Exception:
    doc_col = None


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ResearchReport(BaseModel):
    question:     str
    key_findings: list[str] = Field(min_length=2, max_length=5)
    conclusion:   str
    confidence:   Literal["high", "medium", "low"]
    caveats:      list[str]


SUBMIT_REPORT_TOOL = {
    "name": "submit_report",
    "description": "Submit the final structured research report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "key_findings": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
            "conclusion":   {"type": "string"},
            "confidence":   {"type": "string", "enum": ["high", "medium", "low"]},
            "caveats":      {"type": "array", "items": {"type": "string"}},
        },
        "required": ["key_findings", "conclusion", "confidence", "caveats"],
    },
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    question:     str
    query_plan:   list[str]
    raw_findings: str
    analysis:     str
    confidence:   str
    approved:     bool
    human_note:   str
    report:       dict


# ---------------------------------------------------------------------------
# Tools for researcher
# ---------------------------------------------------------------------------

def search_docs(query: str) -> dict:
    if doc_col is None:
        return {"result": "Doc store not available. Using Claude knowledge only."}
    vec     = st_model.encode(query, convert_to_numpy=True).tolist()
    results = doc_col.query(query_embeddings=[vec], n_results=3)
    return {"chunks": results["documents"][0]}


def get_weather(city: str) -> dict:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=5,
        ).json()
        if not geo.get("results"):
            return {"error": f"City not found: {city}"}
        loc = geo["results"][0]
        cw  = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": loc["latitude"], "longitude": loc["longitude"], "current_weather": True},
            timeout=5,
        ).json()["current_weather"]
        return {"city": city, "temp_celsius": cw["temperature"], "wind_kph": cw["windspeed"]}
    except Exception as e:
        return {"error": str(e)}


def calculate(expression: str) -> dict:
    safe_chars = set("0123456789+-*/(). eE")
    if not all(c in safe_chars for c in expression):
        return {"error": "unsafe expression"}
    try:
        result = eval(expression, {"__builtins__": {}, "math": _math})  # noqa
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


TOOL_REGISTRY = {"search_docs": search_docs, "get_weather": get_weather, "calculate": calculate}

TOOL_SCHEMAS = [
    {"name": "search_docs", "description": "Search the local Agentic AI document corpus.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_weather", "description": "Get current weather for a city.",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "calculate", "description": "Evaluate a safe math expression.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: ResearchState) -> dict:
    """Generate a 3-step research plan."""
    resp = claude.messages.create(
        model=MODEL, max_tokens=256,
        system="Output a JSON array of exactly 3 research steps. Return ONLY the JSON array.",
        messages=[{"role": "user", "content": f"Question: {state['question']}"}],
    )
    try:
        plan = json.loads(resp.content[0].text.strip())
    except Exception:
        plan = [f"Research: {state['question']}"]
    print(f"\n  [planner] {len(plan)} steps")
    return {"query_plan": plan}


def researcher_node(state: ResearchState) -> dict:
    """Run tool calls to gather findings."""
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["query_plan"]))
    messages   = [{"role": "user", "content":
                   f"Question: {state['question']}\n\nResearch steps:\n{steps_text}\n\nUse tools to gather information."}]
    findings   = ""

    for _ in range(8):
        resp = claude.messages.create(
            model=MODEL, max_tokens=1024,
            system="You are a research assistant. Use tools to gather data. When done, summarise your findings.",
            tools=TOOL_SCHEMAS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            for block in resp.content:
                if hasattr(block, "text"):
                    findings = block.text.strip()
            break

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    fn = TOOL_REGISTRY.get(block.name)
                    result = fn(**block.input) if fn else {"error": "unknown tool"}
                    print(f"    [tool] {block.name}({json.dumps(block.input)[:60]}) → {str(result)[:80]}")
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
            messages.append({"role": "user", "content": tool_results})

    print(f"\n  [researcher] findings length: {len(findings)} chars")
    return {"raw_findings": findings}


def analyst_node(state: ResearchState) -> dict:
    """Analyse findings and assign a confidence level."""
    resp = claude.messages.create(
        model=MODEL, max_tokens=512,
        system=(
            "You are a research analyst. Synthesise the findings into a clear analysis. "
            "End with exactly one line: 'CONFIDENCE: high' or 'CONFIDENCE: low'."
        ),
        messages=[{"role": "user", "content":
                   f"Question: {state['question']}\n\nFindings:\n{state['raw_findings']}"}],
    )
    text  = resp.content[0].text.strip()
    lines = text.split("\n")
    conf  = "low"
    for line in reversed(lines):
        if "CONFIDENCE:" in line.upper():
            conf = "high" if "high" in line.lower() else "low"
            break
    analysis = "\n".join(l for l in lines if "CONFIDENCE:" not in l.upper()).strip()
    print(f"\n  [analyst] confidence: {conf}, analysis length: {len(analysis)} chars")
    return {"analysis": analysis, "confidence": conf}


def reporter_node(state: ResearchState) -> dict:
    """Write the final structured report via submit_report tool."""
    if not state.get("approved", False):
        note = state.get("human_note", "no reason")
        print(f"\n  [reporter] SKIPPED — not approved: {note}")
        return {"report": {"status": "rejected", "reason": note}}

    content = (
        f"Question: {state['question']}\n\n"
        f"Analysis:\n{state['analysis']}\n\n"
        "Call submit_report with the structured output."
    )
    resp = claude.messages.create(
        model=MODEL, max_tokens=512,
        system="Fill and submit the research report using submit_report.",
        tools=[SUBMIT_REPORT_TOOL],
        messages=[{"role": "user", "content": content}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "submit_report":
            try:
                report_obj = ResearchReport(question=state["question"], **block.input)
                print(f"\n  [reporter] report structured successfully")
                return {"report": report_obj.model_dump()}
            except ValidationError as e:
                print(f"\n  [reporter] validation error: {e}")

    return {"report": {"question": state["question"], "conclusion": state["analysis"][:200], "confidence": state["confidence"]}}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def route_after_approval(state: ResearchState) -> Literal["reporter", "__end__"]:
    return "reporter" if state.get("approved", False) else "__end__"


def build_app(memory: MemorySaver):
    g = StateGraph(ResearchState)

    g.add_node("planner",    planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("analyst",    analyst_node)
    g.add_node("reporter",   reporter_node)

    g.add_edge(START,        "planner")
    g.add_edge("planner",    "researcher")
    g.add_edge("researcher", "analyst")
    g.add_conditional_edges("analyst", route_after_approval,
                            {"reporter": "reporter", "__end__": END})
    g.add_edge("reporter",   END)

    return g.compile(
        checkpointer=memory,
        interrupt_after=["analyst"],    # pause AFTER analyst, before reporter
    )


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------

def run_research(question: str, human_action: dict, thread_id: str) -> None:
    print(f"\n{'='*64}")
    print(f"Question: {question}")
    print(f"{'='*64}")

    memory = MemorySaver()
    config = {"configurable": {"thread_id": thread_id}}
    app    = build_app(memory)

    initial: ResearchState = {
        "question": question, "query_plan": [], "raw_findings": "",
        "analysis": "", "confidence": "", "approved": False,
        "human_note": "", "report": {},
    }

    # Phase 1: run to interrupt (after analyst)
    print("\n[Phase 1] Running to interrupt point...")
    app.invoke(initial, config=config)

    # Inspect paused state
    paused = app.get_state(config)
    print(f"\n[PAUSED — awaiting human review]")
    print(f"  Analysis: {paused.values.get('analysis','')[:300]}")
    print(f"  Confidence: {paused.values.get('confidence','')}")

    # Phase 2: human action
    print(f"\n[Human action]: {human_action}")
    app.update_state(config, human_action)

    # Phase 3: resume
    print("\n[Phase 3] Resuming after human decision...")
    final = app.invoke(None, config)

    report = final.get("report", {})
    print(f"\n[FINAL REPORT]")
    for k, v in report.items():
        print(f"  {k}: {str(v)[:200]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    run_research(
        question="What is convergence theory in conciousness point of view? Can you provide a structured report with key findings, conclusion, confidence level, and caveats?",
        human_action={"approved": True, "human_note": "LGTM — good analysis"},
        thread_id="research-1",
    )

   
   
