"""
Day 31 — LangGraph Fundamentals
Week 7: LangGraph | 90-Day Agentic AI Plan

A 3-node graph: planner → executor → reporter
TypedDict state flows through every node.
Compile, invoke, print traces.
"""

import os
import anthropic
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

claude = anthropic.Anthropic()
MODEL  = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    query:    str
    plan:     list[str]   # research steps from planner
    findings: str         # raw content from executor
    report:   str         # polished report from reporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(system: str, user: str, max_tokens: int = 512) -> str:
    resp = claude.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: ResearchState) -> dict:
    """Generate a 3-step research plan for the query."""
    print(f"\n  [planner] planning for: {state['query']}")
    raw = _call(
        system=(
            "You are a research planner. Given a question, output a JSON array of "
            "exactly 3 research steps as strings. Return ONLY the JSON array, "
            "no markdown, no explanation."
        ),
        user=f"Question: {state['query']}",
    )
    import json
    try:
        plan = json.loads(raw)
        if not isinstance(plan, list):
            plan = [raw]
    except Exception:
        plan = [f"Step: {raw[:100]}"]
    print(f"  [planner] {len(plan)} steps:")
    for i, step in enumerate(plan, 1):
        print(f"    {i}. {step}")
    return {"plan": plan}


def executor_node(state: ResearchState) -> dict:
    """Execute each research step and compile findings."""
    print(f"\n  [executor] executing {len(state['plan'])} steps...")
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["plan"]))
    findings = _call(
        system=(
            "You are a research assistant. Given a question and a list of research steps, "
            "execute each step and provide specific, factual findings. "
            "Be concrete. No vague generalisations."
        ),
        user=f"Question: {state['query']}\n\nResearch steps:\n{steps_text}",
        max_tokens=768,
    )
    print(f"  [executor] findings length: {len(findings)} chars")
    return {"findings": findings}


def reporter_node(state: ResearchState) -> dict:
    """Write a polished report from the findings."""
    print(f"\n  [reporter] writing report...")
    report = _call(
        system=(
            "You are a technical writer. Synthesise the raw findings into a clear, "
            "well-structured report answering the original question. "
            "2-3 paragraphs. Professional but direct. No bullet points."
        ),
        user=(
            f"Question: {state['query']}\n\n"
            f"Raw findings:\n{state['findings']}"
        ),
        max_tokens=512,
    )
    print(f"  [reporter] report length: {len(report)} chars")
    return {"report": report}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> object:
    g = StateGraph(ResearchState)

    g.add_node("planner",  planner_node)
    g.add_node("executor", executor_node)
    g.add_node("reporter", reporter_node)

    g.add_edge(START,      "planner")
    g.add_edge("planner",  "executor")
    g.add_edge("executor", "reporter")
    g.add_edge("reporter", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

QUERIES = [
    "What is consciousness? Will AI ever be conscious?",
]

if __name__ == "__main__":
    app = build_graph()

    # Optional: print the graph structure
    print("\nGraph structure:")
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        print("  planner → executor → reporter")

    for query in QUERIES:
        print(f"\n{'='*64}")
        print(f"Query: {query}")
        print(f"{'='*64}")

        initial_state: ResearchState = {
            "query":    query,
            "plan":     [],
            "findings": "",
            "report":   "",
        }

        final_state = app.invoke(initial_state)

        print(f"\n  Plan ({len(final_state['plan'])} steps):")
        for i, step in enumerate(final_state["plan"], 1):
            print(f"    {i}. {step}")
        print(f"\n  Report:\n  {final_state['report'][:400]}...")
