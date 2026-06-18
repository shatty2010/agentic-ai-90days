"""
Day 32 — State Management in LangGraph
Week 7: LangGraph | 90-Day Agentic AI Plan

Demonstrates:
  - Annotated reducers (list accumulation vs last-write-wins)
  - MemorySaver checkpointing
  - Thread IDs for independent concurrent conversations
"""

import os
import operator
import json
import anthropic
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

claude = anthropic.Anthropic()
MODEL  = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ConversationState(TypedDict):
    thread_id:  str
    messages:   Annotated[list[str], operator.add]   # accumulates — never replaced
    tool_calls: Annotated[list[str], operator.add]   # tracks tools used across turns
    turn_count: int                                   # last-write-wins
    summary:    str                                   # latest summary


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def respond_node(state: ConversationState) -> dict:
    """Generate a response to the latest user message and track turn count."""
    latest  = state["messages"][-1] if state["messages"] else "Hello"
    history = "\n".join(f"- {m}" for m in state["messages"][:-1])

    system = (
        "You are a helpful AI assistant. Answer the question concisely (2-3 sentences). "
        "If there is conversation history, acknowledge it briefly."
    )
    user = f"Conversation history:\n{history}\n\nCurrent question: {latest}" if history else latest

    resp = claude.messages.create(
        model=MODEL,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    answer = resp.content[0].text.strip()

    new_turn = state.get("turn_count", 0) + 1
    print(f"  [respond] turn {new_turn} — response length: {len(answer)} chars")

    return {
        "messages":   [f"Assistant: {answer}"],   # appended via reducer
        "turn_count": new_turn,
    }


def summarise_node(state: ConversationState) -> dict:
    """Write a one-sentence running summary of the conversation so far."""
    all_msgs = "\n".join(state["messages"])
    resp = claude.messages.create(
        model=MODEL,
        max_tokens=128,
        system="Write a single sentence summarising this conversation. No preamble.",
        messages=[{"role": "user", "content": all_msgs}],
    )
    summary = resp.content[0].text.strip()
    print(f"  [summarise] summary: {summary[:100]}")
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    g = StateGraph(ConversationState)

    g.add_node("respond",   respond_node)
    g.add_node("summarise", summarise_node)

    g.add_edge(START,       "respond")
    g.add_edge("respond",   "summarise")
    g.add_edge("summarise", END)

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_state(label: str, state: dict) -> None:
    print(f"\n  ── {label} ──")
    print(f"  thread_id:  {state.get('thread_id', '-')}")
    print(f"  turn_count: {state.get('turn_count', 0)}")
    print(f"  messages ({len(state.get('messages', []))}):")
    for m in state.get("messages", []):
        print(f"    {m[:200]}")
    print(f"  summary:    {state.get('summary', '')[:200]}")


def run_turn(app, thread_id: str, user_message: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(
        {"thread_id": thread_id, "messages": [f"User: {user_message}"]},
        config=config,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    memory = MemorySaver()
    app    = build_graph(checkpointer=memory)

    # --- Demo 1: multi-turn conversation for user-1 ---
    print("\n" + "="*64)
    print("DEMO 1 — multi-turn conversation (thread: user-1)")
    print("="*64)

    turns_user1 = [
        "What is Radio active elements? Give me 3 examples.",
    ]
    for msg in turns_user1:
        print(f"\n  User: {msg}")
        state = run_turn(app, "user-1", msg)

    print_state("user-1 final state", state)

    # --- Demo 2: independent conversation for user-2 ---
    print("\n" + "="*64)
    print("DEMO 2 — independent conversation (thread: user-2)")
    print("="*64)

    turns_user2 = [
        "What is RAG?",
        "What is the difference between dense and sparse retrieval?",
    ]
    for msg in turns_user2:
        print(f"\n  User: {msg}")
        state2 = run_turn(app, "user-2", msg)

    print_state("user-2 final state", state2)

    # --- Demo 3: reducer proof ---
    print("\n" + "="*64)
    print("DEMO 3 — reducer proof: user-1 has 6 messages, user-2 has 4")
    print("="*64)

    u1_state = app.get_state({"configurable": {"thread_id": "user-1"}})
    u2_state = app.get_state({"configurable": {"thread_id": "user-2"}})

    print(f"  user-1 message count: {len(u1_state.values.get('messages', []))}")
    print(f"  user-2 message count: {len(u2_state.values.get('messages', []))}")
    print(f"  user-1 turn count:    {u1_state.values.get('turn_count', 0)}")
    print(f"  user-2 turn count:    {u2_state.values.get('turn_count', 0)}")
    print("\n  States are independent — MemorySaver isolates by thread_id.")
