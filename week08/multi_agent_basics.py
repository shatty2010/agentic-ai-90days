"""
Day 36 — Multi-Agent Fundamentals
Week 8: Multi-Agent Systems | 90-Day Agentic AI Plan

Two patterns:
  1. Agent-as-tool  — supervisor treats a full worker agent as a callable tool
  2. Supervisor/worker — explicit task decomposition + agent dispatch
"""

import os
import json
import math as _math
import requests
import anthropic
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

claude = anthropic.Anthropic()
MODEL  = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Shared tools (used by worker agents)
# ---------------------------------------------------------------------------

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
            params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                    "current_weather": True}, timeout=5,
        ).json()["current_weather"]
        return {"city": city, "temp_celsius": cw["temperature"], "wind_kph": cw["windspeed"]}
    except Exception as e:
        return {"error": str(e)}


def calculate(expression: str) -> dict:
    if not all(c in set("0123456789+-*/(). eE") for c in expression):
        return {"error": "unsafe expression"}
    try:
        return {"expression": expression,
                "result": eval(expression, {"__builtins__": {}, "math": _math})}  # noqa
    except Exception as e:
        return {"error": str(e)}


TOOL_SCHEMAS = [
    {"name": "get_weather", "description": "Get current weather for a city.",
     "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "calculate", "description": "Evaluate a safe math expression.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
]
TOOL_REGISTRY = {"get_weather": get_weather, "calculate": calculate}


def _tool_loop(system: str, user: str, max_steps: int = 6) -> str:
    """Run a Claude call with tool dispatch. Returns final text."""
    messages = [{"role": "user", "content": user}]
    for _ in range(max_steps):
        resp = claude.messages.create(
            model=MODEL, max_tokens=512, system=system,
            tools=TOOL_SCHEMAS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "end_turn":
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text.strip()
            return ""
        if resp.stop_reason == "tool_use":
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    fn = TOOL_REGISTRY.get(block.name)
                    r  = fn(**block.input) if fn else {"error": "unknown"}
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(r)})
            messages.append({"role": "user", "content": results})
    return "Max steps reached."


# ============================================================================
# Pattern 1 — Agent-as-tool
# ============================================================================

def researcher_worker(query: str) -> str:
    """A complete worker agent. Uses tools to answer a research query."""
    print(f"    [researcher_worker] query: {query[:60]}")
    result = _tool_loop(
        system="You are a research specialist. Use tools to gather facts. Be specific and concise.",
        user=query,
    )
    print(f"    [researcher_worker] done ({len(result)} chars)")
    return result


RESEARCHER_TOOL_SCHEMA = {
    "name": "call_researcher",
    "description": (
        "Delegate a research or data-gathering question to the Researcher agent. "
        "It has access to live weather data and a calculator."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The research question to answer"}},
        "required": ["query"],
    },
}


def run_as_tool_demo(goal: str) -> str:
    """
    Supervisor calls Claude with call_researcher as a tool.
    When Claude calls it, we dispatch to researcher_worker.
    """
    print(f"\n  [Supervisor] goal: {goal[:80]}")
    messages = [{"role": "user", "content": goal}]

    for _ in range(8):
        resp = claude.messages.create(
            model=MODEL, max_tokens=768,
            system=(
                "You are a supervisor agent. Use the call_researcher tool to gather "
                "any data you need, then synthesise a final answer. "
                "Be clear and direct in your final response."
            ),
            tools=[RESEARCHER_TOOL_SCHEMA],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text.strip()
            return ""

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use" and block.name == "call_researcher":
                    print(f"\n  [Supervisor → Researcher] query: {block.input['query'][:60]}")
                    result = researcher_worker(block.input["query"])
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})

    return "Max steps reached."


# ============================================================================
# Pattern 2 — Supervisor/worker with explicit task decomposition
# ============================================================================

class Task(TypedDict):
    task_id: str
    agent:   str    # "researcher" | "analyser" | "writer"
    input:   str
    output:  str


def decompose(goal: str) -> list[Task]:
    """Ask Claude to break the goal into ordered agent tasks."""
    resp = claude.messages.create(
        model=MODEL, max_tokens=512,
        system=(
            "You are a task planner for a multi-agent system with three agents: "
            "researcher (gathers facts/data), analyser (interprets and finds patterns), "
            "writer (produces final polished output). "
            "Break the goal into 2-4 tasks. Each task must have: "
            "task_id (t1, t2...), agent, input (what to do). "
            "Return ONLY a JSON array. No markdown."
        ),
        messages=[{"role": "user", "content": f"Goal: {goal}"}],
    )
    raw = resp.content[0].text.strip()
    try:
        tasks = json.loads(raw)
        return [Task(task_id=t["task_id"], agent=t["agent"],
                     input=t["input"], output="") for t in tasks]
    except Exception:
        return [Task(task_id="t1", agent="researcher", input=goal, output="")]


def researcher_agent(task: Task) -> str:
    return _tool_loop(
        system="You are a Researcher agent. Gather data and facts. Use tools when needed. Be specific.",
        user=task["input"],
    )


def analyser_agent(task: Task, prior_results: dict) -> str:
    context = "\n\n".join(f"[{k}]: {v}" for k, v in prior_results.items())
    return _tool_loop(
        system="You are an Analyser agent. Identify patterns, draw comparisons, and surface insights. Be concise.",
        user=f"Prior research:\n{context}\n\nYour task: {task['input']}",
    )


def writer_agent(task: Task, prior_results: dict) -> str:
    context = "\n\n".join(f"[{k}]: {v}" for k, v in prior_results.items())
    resp = claude.messages.create(
        model=MODEL, max_tokens=512,
        system="You are a Writer agent. Produce clear, well-structured final output. No jargon.",
        messages=[{"role": "user", "content":
                   f"All agent outputs:\n{context}\n\nYour task: {task['input']}"}],
    )
    return resp.content[0].text.strip()


def run_supervisor(goal: str) -> str:
    print(f"\n{'='*64}")
    print(f"[Supervisor] Goal: {goal}")
    print(f"{'='*64}")

    tasks        = decompose(goal)
    prior_results: dict[str, str] = {}

    print(f"\n  [Decompose] {len(tasks)} tasks:")
    for t in tasks:
        print(f"    {t['task_id']} → {t['agent']}: {t['input'][:60]}")

    for task in tasks:
        print(f"\n  [Dispatch] {task['task_id']} → {task['agent']}")
        if task["agent"] == "researcher":
            output = researcher_agent(task)
        elif task["agent"] == "analyser":
            output = analyser_agent(task, prior_results)
        elif task["agent"] == "writer":
            output = writer_agent(task, prior_results)
        else:
            output = f"Unknown agent: {task['agent']}"

        task["output"]  = output
        prior_results[task["task_id"]] = output
        print(f"  [Done] {task['task_id']}: {output[:100]}...")

    # Final answer = last task's output (writer)
    return tasks[-1]["output"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GOALS = [
    (
        "Research the current weather in Paris and Barcelona, analyse which city is "
        "better for a tech conference in July, and write a 3-sentence recommendation.",
        "agent-as-tool",
    )
]

if __name__ == "__main__":
    for goal, pattern in GOALS:
        print(f"\n{'#'*64}")
        print(f"Pattern: {pattern}")
        print(f"{'#'*64}")

        if pattern == "agent-as-tool":
            final = run_as_tool_demo(goal)
        else:
            final = run_supervisor(goal)

        print(f"\n  FINAL ANSWER:\n  {final[:400]}")
