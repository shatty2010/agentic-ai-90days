"""
Day 27 — Plan-and-Execute Pattern
Week 6: Agent Architectures | 90-Day Agentic AI Plan

Two-phase agent:
  Phase 1 (Planner)    — receives goal, produces numbered JSON plan, no tools
  Phase 2 (Executor)   — runs each step with tools, accumulates results
  Phase 3 (Synthesiser)— combines all step results into a final answer
"""

import os
import re
import json
import time
import math
import requests
import anthropic
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Tools (same as Day 26)
# ---------------------------------------------------------------------------

def get_weather(city: str) -> dict:
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


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    resp = requests.get(
        "https://api.frankfurter.app/latest",
        params={"amount": amount, "from": from_currency.upper(), "to": to_currency.upper()},
        timeout=5,
    ).json()
    if "rates" not in resp:
        return {"error": resp.get("message", "failed")}
    result = resp["rates"][to_currency.upper()]
    return {"amount": amount, "from": from_currency.upper(), "to": to_currency.upper(),
            "result": result, "rate": round(result / amount, 6)}


def calculate(expression: str) -> dict:
    safe_globals = {"__builtins__": {}, "math": math}
    if not all(c in set("0123456789+-*/(). eE") for c in expression):
        return {"error": f"Unsafe expression: {expression}"}
    try:
        return {"expression": expression, "result": eval(expression, safe_globals)}  # noqa: S307
    except Exception as exc:
        return {"error": str(exc)}


TOOL_REGISTRY = {
    "get_weather":      get_weather,
    "convert_currency": convert_currency,
    "calculate":        calculate,
}

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "convert_currency",
        "description": "Convert between currencies using live ECB rates.",
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
        "description": "Evaluate a mathematical expression.",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
]


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _safe_call(name: str, fn, **kwargs) -> dict:
    try:
        return fn(**kwargs)
    except Exception as exc:
        return {"error": f"{name}: {exc}"}


def _run_tool_loop(messages: list, system: str, max_steps: int = 5) -> str:
    """Mini agent loop for the executor — runs until end_turn."""
    for _ in range(max_steps):
        resp = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # Dispatch tool calls
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                fn     = TOOL_REGISTRY.get(block.name)
                result = _safe_call(block.name, fn, **block.input) if fn else {"error": f"Unknown: {block.name}"}
                print(f"    Tool: {block.name}({block.input}) → {result}")
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        messages.append({"role": "user", "content": tool_results})

    return "Max steps reached in executor."


# ---------------------------------------------------------------------------
# Three phases
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """You are a planning assistant.

Given the user's goal, produce a numbered list of concrete, self-contained steps.
Return ONLY a valid JSON array of strings. No explanation, no markdown.

Example:
["Step 1: Get the current weather in London", "Step 2: Get the current weather in Tokyo", "Step 3: Compare the two temperatures"]"""


SYNTHESISER_PROMPT = """You are a synthesis assistant.
Given the original goal, the execution plan, and the result of each step,
write a single coherent answer that addresses the goal completely.
Be concise. Do not repeat the plan. Just answer the goal."""


def plan(goal: str) -> list[str]:
    """Phase 1: produce a step-by-step JSON plan."""
    resp = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=PLANNER_PROMPT,
        messages=[{"role": "user", "content": f"Goal: {goal}"}],
    )
    raw = resp.content[0].text.strip()
    # Strip markdown code fences the model occasionally adds despite instructions
    raw = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip()
    try:
        steps = json.loads(raw)
        if isinstance(steps, list):
            return steps
    except json.JSONDecodeError:
        pass
    # Fallback: split on newlines if JSON parse fails
    return [line.strip() for line in raw.splitlines() if line.strip()]


def execute_step(step: str, step_num: int, previous_results: list[str]) -> str:
    """Phase 2: execute one step using tools. Returns result as text."""
    prior = "\n".join(
        f"Step {i+1} result: {r}" for i, r in enumerate(previous_results)
    ) or "None yet."

    system = (
        f"Execute ONLY this step: {step}\n\n"
        f"Previous step results:\n{prior}\n\n"
        "Use tools if you need live data. Return a clear, concise result for this step only."
    )
    messages = [{"role": "user", "content": step}]
    print(f"\n  Executing Step {step_num}: {step}")
    result = _run_tool_loop(messages, system)
    print(f"  Result: {result[:120]}...")
    return result


def synthesise(goal: str, steps: list[str], results: list[str]) -> str:
    """Phase 3: combine all step results into one final answer."""
    plan_text    = "\n".join(f"  {s}" for s in steps)
    results_text = "\n".join(f"  Step {i+1}: {r}" for i, r in enumerate(results))
    content      = f"Goal: {goal}\n\nPlan:\n{plan_text}\n\nStep results:\n{results_text}"

    resp = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=SYNTHESISER_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_plan_execute(goal: str) -> str:
    print(f"\n{'='*64}")
    print(f"Goal: {goal}")
    print(f"{'='*64}")

    # Phase 1 — Plan
    steps = plan(goal)
    print(f"\nPlan ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    # Phase 2 — Execute
    results = []
    for i, step in enumerate(steps, 1):
        result = execute_step(step, i, results)
        results.append(result)

    # Phase 3 — Synthesise
    answer = synthesise(goal, steps, results)
    return answer


GOALS = [
    "Get the weather in Berlin and Singapore, compare them, then convert 1000 EUR to SGD.",
    "What is 15% of 4827, what is 2 to the power of 15, and which result is larger?",
    "Get the weather in London, find out how warm Tokyo is, and tell me which city is warmer and by how much in Fahrenheit (multiply Celsius difference by 1.8).",
]

if __name__ == "__main__":
    for goal in GOALS:
        answer = run_plan_execute(goal)
        print(f"\n  Final Answer:\n  {answer}\n")
