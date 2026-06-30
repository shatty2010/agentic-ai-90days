"""
Day 34 — Human-in-the-Loop (HITL)
Week 7: LangGraph | 90-Day Agentic AI Plan

Demonstrates:
  - interrupt_before to pause execution before a "review" node
  - MemorySaver checkpointing so state survives the pause
  - app.get_state to inspect the paused state
  - app.update_state to let the human edit the draft
  - app.invoke(None, config) to resume

Three scenarios: approve, edit-then-approve, reject.
"""

import os
import anthropic
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

claude = anthropic.Anthropic()
MODEL  = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ReviewState(TypedDict):
    task:         str
    draft:        str    # agent-generated action plan
    human_note:   str    # reviewer comment or edit reason
    approved:     bool
    final_output: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def draft_node(state: ReviewState) -> dict:
    """Generate a draft action plan for the given task."""
    resp = claude.messages.create(
        model=MODEL,
        max_tokens=512,
        system=(
            "You are a project planning assistant. "
            "Given a task, write a clear 3-step action plan. "
            "Each step should be one sentence. "
            "Format: '1. ... 2. ... 3. ...'"
        ),
        messages=[{"role": "user", "content": f"Task: {state['task']}"}],
    )
    draft = resp.content[0].text.strip()
    print(f"\n  [draft_node] plan generated ({len(draft)} chars):")
    print(f"  {draft[:300]}")
    return {"draft": draft, "approved": False}


def review_node(state: ReviewState) -> dict:
    """
    This node represents the human reviewer's decision.
    In production: this node reads state['draft'], shows it in a UI,
    and the human's action (approve/edit/reject) updates the state before this
    node runs.
    In this script: the node just marks what a human would see and stamps a
    default 'reviewed' note — actual human decisions are simulated via
    app.update_state() BEFORE resume (so this node just validates the flag).
    """
    approved = state.get("approved", False)
    note     = state.get("human_note", "")
    print(f"\n  [review_node] approved={approved}, note='{note}'")
    # No state change — human already set approved + human_note via update_state
    return {}


def execute_node(state: ReviewState) -> dict:
    """Execute the approved plan. Skip if rejected."""
    if not state.get("approved", False):
        reason = state.get("human_note", "no reason given")
        print(f"\n  [execute_node] SKIPPED — rejected: {reason}")
        return {"final_output": f"Not executed. Rejection reason: {reason}"}

    draft = state["draft"]
    resp = claude.messages.create(
        model=MODEL,
        max_tokens=512,
        system=(
            "You are an execution report writer. "
            "Given an action plan, write a short execution confirmation: "
            "what was done, what the outcome was. 2-3 sentences."
        ),
        messages=[{"role": "user", "content": f"Executed plan:\n{draft}"}],
    )
    output = resp.content[0].text.strip()
    print(f"\n  [execute_node] execution complete:")
    print(f"  {output[:200]}")
    return {"final_output": output}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_app(thread_id: str, memory: MemorySaver):
    g = StateGraph(ReviewState)

    g.add_node("draft",   draft_node)
    g.add_node("review",  review_node)
    g.add_node("execute", execute_node)

    g.add_edge(START,     "draft")
    g.add_edge("draft",   "review")
    g.add_edge("review",  "execute")
    g.add_edge("execute", END)

    return g.compile(
        checkpointer=memory,
        interrupt_before=["review"],   # pause BEFORE the review node
    )


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def run_scenario(scenario_name: str, task: str, human_action: dict | None) -> None:
    print(f"\n{'='*64}")
    print(f"SCENARIO: {scenario_name}")
    print(f"Task: {task}")
    print(f"{'='*64}")

    memory    = MemorySaver()
    thread_id = f"scenario-{scenario_name.lower().replace(' ', '-')}"
    config    = {"configurable": {"thread_id": thread_id}}
    app       = build_app(thread_id, memory)

    # Phase 1: run until interrupt
    print("\n  [Phase 1] Running graph to interrupt point...")
    app.invoke(
        {"task": task, "draft": "", "human_note": "", "approved": False, "final_output": ""},
        config=config,
    )

    # Inspect paused state
    paused = app.get_state(config)
    print(f"\n  [PAUSED] State at interrupt:")
    print(f"  draft:  {paused.values.get('draft', '')[:200]}")
    print(f"  next:   {paused.next}")

    # Phase 2: human action
    if human_action is not None:
        print(f"\n  [Human action] {human_action}")
        app.update_state(config, human_action)

    # Phase 3: resume
    print("\n  [Phase 3] Resuming graph...")
    final = app.invoke(None, config)

    print(f"\n  [FINAL STATE]")
    print(f"  approved:     {final.get('approved')}")
    print(f"  human_note:   {final.get('human_note', '')}")
    print(f"  final_output: {final.get('final_output', '')[:200]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # # Scenario 1 — human approves as-is
    # run_scenario(
    #     scenario_name="Approve",
    #     task="Set up a CI/CD pipeline for our new Python microservice",
    #     human_action={"approved": True, "human_note": "LGTM"},
    # )

    # # Scenario 2 — human edits draft then approves
    # run_scenario(
    #     scenario_name="Edit then Approve",
    #     task="Migrate our database from MySQL to PostgreSQL",
    #     human_action={
    #         "draft":      "1. Run pg_dump on MySQL schema. 2. Set up PostgreSQL instance on RDS. 3. Migrate data with pgloader.",
    #         "human_note": "Edited: added pgloader as migration tool",
    #         "approved":   True,
    #     },
    # )

    # Scenario 3 — human rejects
    run_scenario(
        scenario_name="Reject",
        task="Delete all inactive user accounts older than 2 years",
        human_action={
            "approved":   False,
            "human_note": "Rejected: needs legal review before any user data deletion",
        },
    )
