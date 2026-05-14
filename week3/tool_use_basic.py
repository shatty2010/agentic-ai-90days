"""
Day 11 — Claude Native Tool Use
--------------------------------
Goal: Understand the tool_use API loop end-to-end.

Tools are hardcoded (no real HTTP calls yet — that's Day 12).
Focus: the protocol. tool_use content block → dispatch → tool_result injection.

Run:
    pip install anthropic
    export ANTHROPIC_API_KEY=your_key   (or set in .env)
    python tool_use_basic.py
"""

import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Tool schemas ──────────────────────────────────────────────────────────────
# These are what Claude reads to understand what tools exist and how to call them.
# The description field is critical — it tells Claude WHEN to use this tool.
# The input_schema is a JSON Schema object — same format used in OpenAPI.

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Use when the user asks about weather, temperature, or conditions in a specific place. "
            "Do NOT use for historical weather or forecasts — only current conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g. 'London', 'Tokyo'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_population",
        "description": (
            "Get the approximate population of a country. "
            "Use when the user asks how many people live in a country. "
            "This returns approximate figures, not census-exact data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "The name of the country, e.g. 'India', 'Germany'"
                }
            },
            "required": ["country"]
        }
    }
]

# ── Tool implementations ──────────────────────────────────────────────────────
# Hardcoded for Day 11. Day 12 replaces these with real HTTP calls.

def get_weather(city: str) -> dict:
    data = {
        "London":   {"temp": "18°C", "condition": "Cloudy"},
        "Tokyo":    {"temp": "26°C", "condition": "Sunny"},
        "Sydney":   {"temp": "14°C", "condition": "Rainy"},
        "New York": {"temp": "22°C", "condition": "Clear"},
        "Mumbai":   {"temp": "33°C", "condition": "Humid"},
        "Berlin":   {"temp": "16°C", "condition": "Overcast"},
        "Singapore":{"temp": "31°C", "condition": "Partly Cloudy"},
    }
    return data.get(city, {"temp": "N/A", "condition": "City not in dataset"})


def get_population(country: str) -> dict:
    data = {
        "India":     {"population": "1.44 billion"},
        "China":     {"population": "1.41 billion"},
        "USA":       {"population": "335 million"},
        "Australia": {"population": "26 million"},
        "Germany":   {"population": "84 million"},
        "Singapore": {"population": "6 million"},
        "Japan":     {"population": "124 million"},
    }
    return data.get(country, {"population": "N/A — country not in dataset"})


TOOL_REGISTRY = {
    "get_weather":   get_weather,
    "get_population": get_population,
}

# ── Core loop ─────────────────────────────────────────────────────────────────

def dispatch_tool(tool_use_block) -> dict:
    """
    Execute the tool Claude requested.
    Returns a dict result — always. Never raises.
    """
    name  = tool_use_block.name
    args  = tool_use_block.input
    fn    = TOOL_REGISTRY.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    return fn(**args)


def run_agent(question: str) -> str:
    """
    Full tool_use agent loop.

    Messages flow:
      user: question
      assistant: [tool_use block]       ← Claude wants to call a tool
      user: [tool_result block]         ← we give it the result
      assistant: final text answer      ← stop_reason == 'end_turn'
    """
    print(f"\n{'='*64}")
    print(f"  Q: {question}")
    print(f"{'='*64}")

    messages = [{"role": "user", "content": question}]

    step = 0
    while True:
        step += 1
        print(f"\n[Step {step}] → Calling Claude")

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",   # fast + cheap for practice
            max_tokens=1024,
            tools=TOOL_SCHEMAS,
            messages=messages
        )

        print(f"           stop_reason: {response.stop_reason}")

        # ── Claude is done — return its text answer ──────────────────────────
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\n  Answer: {block.text}")
                    return block.text
            return "[No text in response]"

        # ── Claude wants to call tools ────────────────────────────────────────
        if response.stop_reason == "tool_use":
            # Step 1: append Claude's response (contains the tool_use block)
            messages.append({"role": "assistant", "content": response.content})

            # Step 2: dispatch every tool_use block in the response
            # (Claude can request multiple tools in one turn — handle all of them)
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"           tool call: {block.name}({block.input})")
                    result = dispatch_tool(block)
                    print(f"           result:    {result}")

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,          # must match Claude's request ID
                        "content":     json.dumps(result)
                    })

            # Step 3: inject ALL results in one user turn
            messages.append({"role": "user", "content": tool_results})

            # Loop continues — Claude will now read the results and respond

        else:
            # Unexpected stop reason — exit cleanly
            print(f"  [!] Unexpected stop_reason: {response.stop_reason}")
            return f"Loop ended: {response.stop_reason}"


# ── Tests ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        # Needs get_weather only
        "What is the weather in a city that doesn't exist called Blorbtown?",

        # Needs get_population only
        "What is the population of Mars?",

        # Needs neither — Claude should answer from its own knowledge
        "Who was the first Prime Minister of Australia?",
    ]

    for q in test_questions:
        run_agent(q)