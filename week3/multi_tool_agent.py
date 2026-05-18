"""
Day 13 — Multi-Tool Orchestration
-----------------------------------
Goal: 4-tool agent. Handle parallel tool calls. Learn how tool description
quality controls which tool Claude selects.

Tools:
  • search_wikipedia(query)                        — from Day 12
  • get_weather(city)                              — from Day 12
  • convert_currency(amount, from_currency, to_currency)  — Frankfurter API (free, no key)
  • calculate(expression)                          — eval() with safety wrapper

Key changes from Day 12:
  • dispatch_all() handles multiple tool_use blocks in one response (parallel calls)
  • All tool results injected in a single user turn — required by the API
  • Step logger prints which tools were called each turn

Run:
    pip install anthropic requests python-dotenv
    export ANTHROPIC_API_KEY=your_key
    python multi_tool_agent.py
"""

import json
import os
import re
from typing import Optional, Tuple

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

TIMEOUT = 5  # seconds

# ── WMO weather codes ─────────────────────────────────────────────────────────

WMO_CODES: dict[int, str] = {
    0:  "Clear sky",     1:  "Mainly clear",   2:  "Partly cloudy",
    3:  "Overcast",      45: "Fog",             48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle",55: "Dense drizzle",
    61: "Slight rain",   63: "Moderate rain",   65: "Heavy rain",
    71: "Slight snow",   73: "Moderate snow",   75: "Heavy snow",
    80: "Slight showers",81: "Moderate showers",82: "Heavy showers",
    95: "Thunderstorm",  96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}

# ── Tool schemas ──────────────────────────────────────────────────────────────
# The description is the most important field.
# It tells Claude WHEN to use a tool, WHEN NOT TO, and what to expect from inputs.
# Compare v1/v2/v3 in the hands-on block to see how much this matters.

TOOL_SCHEMAS = [
    {
        "name": "search_wikipedia",
        "description": (
            "Search Wikipedia for factual background on a topic. "
            "Use when the question asks about historical facts, definitions, people, "
            "places, events, or scientific concepts. "
            "Do NOT use for current prices, live weather, or real-time data — "
            "Wikipedia is not a live source."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The topic to look up — use the most likely Wikipedia article title, "
                        "e.g. 'Eiffel Tower', 'Ada Lovelace', 'Machine learning'"
                    )
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Use when the user asks about weather, temperature, or atmospheric conditions "
            "in a specific place right now. "
            "Do NOT use for historical weather data or multi-day forecasts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Tokyo', 'Berlin', 'Mumbai'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "convert_currency",
        "description": (
            "Convert an amount from one currency to another using live exchange rates. "
            "Use when the user asks about currency conversion, exchange rates, or "
            "how much a foreign amount is worth in another currency. "
            "Use standard 3-letter ISO currency codes: USD, EUR, GBP, INR, AUD, SGD, JPY, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "The amount to convert, e.g. 100"
                },
                "from_currency": {
                    "type": "string",
                    "description": "Source currency ISO code, e.g. 'EUR'"
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency ISO code, e.g. 'USD'"
                }
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression and return the result. "
            "Use when the user asks for arithmetic, percentages, powers, or any numeric computation. "
            "Input must be a valid math expression using numbers and operators only: "
            "+ - * / ** % ( ). No variables, no function names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A safe math expression, e.g. '15 * 3 + 7', '2 ** 10', '(100 / 3) * 1.1'"
                    )
                }
            },
            "required": ["expression"]
        }
    }
]

# ── Tool implementations ───────────────────────────────────────────────────────

def _first_n_sentences(text: str, n: int = 3) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:n])


def search_wikipedia(query: str) -> dict:
    title = query.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers={"User-Agent": "agentic-ai-learner/1.0"})
        if r.status_code == 404:
            return {"error": f"No Wikipedia page found for '{query}'"}
        r.raise_for_status()
        data = r.json()
        extract = data.get("extract", "").strip()
        if not extract:
            return {"error": "Page exists but has no summary text"}
        return {"title": data.get("title", query), "summary": _first_n_sentences(extract, 3)}
    except requests.Timeout:
        return {"error": "Wikipedia request timed out"}
    except requests.HTTPError as e:
        return {"error": f"Wikipedia HTTP error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Wikipedia lookup failed: {str(e)}"}


def _geocode_city(city: str) -> Optional[Tuple[float, float]]:
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=TIMEOUT
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return (results[0]["latitude"], results[0]["longitude"]) if results else None
    except Exception:
        return None


def get_weather(city: str) -> dict:
    coords = _geocode_city(city)
    if coords is None:
        return {"error": f"City not found: '{city}'"}
    lat, lon = coords
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=TIMEOUT
        )
        r.raise_for_status()
        cw = r.json().get("current_weather")
        if not cw:
            return {"error": "Weather data unavailable for this location"}
        return {
            "temp_celsius": cw["temperature"],
            "condition":    WMO_CODES.get(cw.get("weathercode", -1), "Unknown"),
        }
    except requests.Timeout:
        return {"error": "Weather request timed out"}
    except requests.HTTPError as e:
        return {"error": f"Weather API error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Weather lookup failed: {str(e)}"}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Live exchange rates via Frankfurter API (free, no key, backed by ECB data).
    API: GET https://api.frankfurter.app/latest?amount=X&from=EUR&to=USD
    """
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={
                "amount": amount,
                "from":   from_currency.upper(),
                "to":     to_currency.upper(),
            },
            timeout=TIMEOUT
        )
        if r.status_code == 422:
            return {"error": f"Unknown currency code. Use ISO codes like USD, EUR, GBP, INR."}
        r.raise_for_status()
        data = r.json()
        rates = data.get("rates", {})
        target = to_currency.upper()
        if target not in rates:
            return {"error": f"No rate found for {target}"}
        return {
            "result":        rates[target],
            "from_currency": from_currency.upper(),
            "to_currency":   target,
            "rate":          rates[target] / amount,
        }
    except requests.Timeout:
        return {"error": "Currency API timed out"}
    except requests.HTTPError as e:
        return {"error": f"Currency API error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Currency conversion failed: {str(e)}"}


# Whitelist of allowed characters — prevents code injection via eval()
_SAFE_EXPR = re.compile(r'^[\d\s\+\-\*\/\%\(\)\.\*\*]+$')

def calculate(expression: str) -> dict:
    """
    Evaluate a math expression safely.
    Only allows digits, spaces, and math operators. No builtins, no names.
    """
    expr = expression.strip()
    if not _SAFE_EXPR.match(expr):
        return {"error": f"Unsafe expression rejected: '{expr}'. Use numbers and operators only."}
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # no builtins, no globals
        return {"result": result}
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": f"Could not evaluate expression: {str(e)}"}


TOOL_REGISTRY = {
    "search_wikipedia": search_wikipedia,
    "get_weather":      get_weather,
    "convert_currency": convert_currency,
    "calculate":        calculate,
}

# ── Agent loop ────────────────────────────────────────────────────────────────
# Key change from Day 12: dispatch_all() handles the full list of tool_use blocks.
# Claude can request multiple tools in a single response — we run them all,
# collect all results, and inject them in one user turn.
# Sending them as separate turns breaks the API contract.

def dispatch_all(tool_use_blocks: list) -> list[dict]:
    """
    Run every tool Claude requested in this turn.
    Returns a list of tool_result dicts ready to inject as a user message.
    """
    results = []
    for block in tool_use_blocks:
        fn = TOOL_REGISTRY.get(block.name)
        if fn:
            result = fn(**block.input)
        else:
            result = {"error": f"Unknown tool: {block.name}"}

        print(f"     ↳ [{block.name}] input={block.input}  →  {result}")

        results.append({
            "type":        "tool_result",
            "tool_use_id": block.id,
            "content":     json.dumps(result),
        })
    return results


def run_agent(question: str) -> str:
    print(f"\n{'='*64}")
    print(f"  Q: {question}")
    print(f"{'='*64}")

    messages = [{"role": "user", "content": question}]
    step = 0

    while True:
        step += 1
        print(f"\n[Step {step}] Calling Claude...")

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        print(f"           stop_reason: {response.stop_reason}")

        # ── Final answer ──────────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\n  Answer: {block.text}")
                    return block.text
            return "[No text in response]"

        # ── Tool calls requested ──────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            # Collect all tool_use blocks from this response
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            tools_called = [b.name for b in tool_use_blocks]
            print(f"           tools called: {tools_called}")

            # Append Claude's full response (required before injecting results)
            messages.append({"role": "assistant", "content": response.content})

            # Run all tools, collect all results
            tool_results = dispatch_all(tool_use_blocks)

            # Inject ALL results in one user turn — not separate turns
            messages.append({"role": "user", "content": tool_results})

        else:
            print(f"  [!] Unexpected stop_reason: {response.stop_reason}")
            return f"Loop ended: {response.stop_reason}"


# ── Tests ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [

        # 2. Two tools in parallel — Claude should call both in one step
        "What is the weather in XYZNOTACITY and what is 150 USD in JPY right now?",

        # 4. No tool needed — Claude answers from its own knowledge
        "What is 10/0?"

    ]

    for q in test_questions:
        run_agent(q)