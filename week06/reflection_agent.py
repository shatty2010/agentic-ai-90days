"""
Day 28 — Reflection & Self-Critique
Week 6: Agent Architectures | 90-Day Agentic AI Plan

Three-stage loop: Generate → Critique → Refine
Stops when critic severity == "none" or max_iterations reached.
Prints initial vs final answer side by side.
"""

import os
import re
import json
import math
import requests
import anthropic
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Tools
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


def calculate(expression: str) -> dict:
    if not all(c in set("0123456789+-*/(). eE") for c in expression):
        return {"error": "unsafe expression"}
    try:
        return {"expression": expression, "result": eval(expression, {"__builtins__": {}, "math": math})}  # noqa
    except Exception as exc:
        return {"error": str(exc)}


TOOL_REGISTRY = {"get_weather": get_weather, "calculate": calculate}

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "calculate",
        "description": "Evaluate a safe mathematical expression.",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATOR_PROMPT = """You are a research assistant. Answer the question as completely and specifically as possible.
Use tools when you need live data. Be concrete — avoid vague generalisations."""

CRITIC_PROMPT = """You are a strict answer reviewer. Evaluate the answer against the question using this rubric:
  1. Completeness  — does it address every part of the question?
  2. Accuracy      — are all stated facts correct?
  3. Specificity   — does it give concrete details, not vague statements?
  4. Grounding     — is every claim supported by evidence (tool results, known facts)?
  5. Clarity       — is it well structured and readable?

Return ONLY valid JSON (no markdown, no explanation):
{
  "issues": ["specific issue 1", "specific issue 2"],
  "severity": "none" | "minor" | "major"
}

severity "none" means the answer is satisfactory and needs no changes.
severity "minor" means small gaps that should be fixed.
severity "major" means significant problems — incomplete, inaccurate, or unsupported claims."""

REFINER_PROMPT = """You are an answer improvement assistant.
You will receive: a question, a draft answer, and a list of specific issues found by a reviewer.
Produce an improved answer that:
  - Addresses every issue listed
  - Does not introduce unsupported claims
  - Uses tools if you need live data to support a claim
Keep everything that was correct in the original. Only fix what the critique identified."""


# ---------------------------------------------------------------------------
# Three stages
# ---------------------------------------------------------------------------

def _tool_loop(messages: list, system: str, with_tools: bool = True) -> str:
    """Run a Claude call with optional tool dispatch. Returns final text."""
    for _ in range(8):
        resp = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=768,
            system=system,
            tools=TOOL_SCHEMAS if with_tools else [],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text.strip()
            return ""

        if resp.stop_reason == "tool_use" and with_tools:
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    fn     = TOOL_REGISTRY.get(block.name)
                    result = fn(**block.input) if fn else {"error": f"Unknown: {block.name}"}
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
            messages.append({"role": "user", "content": tool_results})

    return "Max steps reached."


def generate(question: str) -> str:
    """Stage 1: produce an initial answer using tools."""
    return _tool_loop(
        messages=[{"role": "user", "content": question}],
        system=GENERATOR_PROMPT,
        with_tools=True,
    )


def critique(question: str, answer: str) -> dict:
    """Stage 2: evaluate the answer. Returns {issues, severity}."""
    content = f"Question: {question}\n\nAnswer to review:\n{answer}"
    resp = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=CRITIC_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip()
    try:
        result = json.loads(raw)
        if "issues" in result and "severity" in result:
            return result
    except json.JSONDecodeError:
        pass
    return {"issues": ["Could not parse critique"], "severity": "minor"}


def refine(question: str, answer: str, critique_result: dict) -> str:
    """Stage 3: improve the answer based on the critique."""
    issues_text = "\n".join(f"  - {issue}" for issue in critique_result.get("issues", []))
    content = (
        f"Question: {question}\n\n"
        f"Draft answer:\n{answer}\n\n"
        f"Issues to fix (severity: {critique_result.get('severity', 'minor')}):\n{issues_text}"
    )
    return _tool_loop(
        messages=[{"role": "user", "content": content}],
        system=REFINER_PROMPT,
        with_tools=True,
    )


# ---------------------------------------------------------------------------
# Reflection loop
# ---------------------------------------------------------------------------

def run_reflection(question: str, max_iterations: int = 3) -> dict:
    print(f"\n{'='*64}")
    print(f"Question: {question}")
    print(f"{'='*64}")

    # Stage 1: generate
    print("\n  [Generate] producing initial answer...")
    initial_answer = generate(question)
    current_answer = initial_answer
    all_critiques  = []

    for iteration in range(1, max_iterations + 1):
        # Stage 2: critique
        print(f"\n  [Critique #{iteration}] reviewing answer...")
        feedback = critique(question, current_answer)
        all_critiques.append(feedback)
        print(f"  Severity: {feedback['severity']}")
        for issue in feedback.get("issues", []):
            print(f"  Issue: {issue}")

        if feedback["severity"] == "none":
            print(f"  Critic satisfied after iteration {iteration}.")
            break

        # Stage 3: refine
        print(f"\n  [Refine #{iteration}] improving answer...")
        current_answer = refine(question, current_answer, feedback)
        print(f"  Refined answer: {current_answer[:200]}...")

    return {
        "question":   question,
        "initial":    initial_answer,
        "final":      current_answer,
        "iterations": len(all_critiques),
        "critiques":  all_critiques,
    }


def print_comparison(result: dict) -> None:
    print(f"\n{'─'*64}")
    print("  INITIAL ANSWER:")
    print(f"  {result['initial']}")
    print(f"\n  FINAL ANSWER (after {result['iterations']} critique round(s)):")
    print(f"  {result['final']}")
    print(f"{'─'*64}")


# ---------------------------------------------------------------------------

QUESTIONS = [
    "What is the weather in Paris today and what should I pack for a day trip there?",
]

if __name__ == "__main__":
    for question in QUESTIONS:
        result = run_reflection(question, max_iterations=3)
        print_comparison(result)
