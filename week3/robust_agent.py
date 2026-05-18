"""
Day 14 — Tool Error Handling & Fallbacks
------------------------------------------
Goal: Harden the agent so it never crashes, even when tools fail.

Key additions over Day 13:
  • safe_call(tool_fn, **kwargs)  — wraps every tool dispatch, catches all exceptions
  • max_retries=1                 — retries once on Timeout only (not on 4xx — those won't fix themselves)
  • tool_call_log                 — list of {tool, input, result, success, duration_ms} per run
  • FORCE_FAIL                    — inject failures per tool for testing without modifying tool functions

Rule: always send a tool_result back to Claude, even on failure.
If you skip it after a tool_use block, the message history becomes invalid.
The error dict IS the tool result.

Run:
    pip install anthropic requests python-dotenv
    export ANTHROPIC_API_KEY=your_key
    python robust_agent.py
"""

import json
import os
import re
import time
from typing import Optional, Tuple

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

TIMEOUT = 5  # seconds

# ── Failure injection ─────────────────────────────────────────────────────────
# Set a tool name here to force it to return a specific error during testing.
# Set to {} for normal operation.
#
# Example — inject a Wikipedia timeout and a currency API failure:
#   FORCE_FAIL = {
#       "search_wikipedia": {"error": "Tool timed out after 5s"},
#       "convert_currency": {"error": "API error 500"},
#   }

FORCE_FAIL: dict[str, dict] = {}

# ── WMO weather codes ─────────────────────────────────────────────────────────

WMO_CODES: dict[int, str] = {
    0:  "Clear sky",      1:  "Mainly clear",    2:  "Partly cloudy",
    3:  "Overcast",       45: "Fog",              48: "Icy fog",
    51: "Light drizzle",  53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain",    63: "Moderate rain",    65: "Heavy rain",
    71: "Slight snow",    73: "Moderate snow",    75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Heavy showers",
    95: "Thunderstorm",   96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}

# ── Tool schemas ───────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "search_wikipedia",
        "description": (
            "Search Wikipedia for factual background on a topic. "
            "Use when the question asks about historical facts, definitions, people, "
            "places, events, or scientific concepts. "
            "Do NOT use for current prices, live weather, or real-time data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic to look up, e.g. 'Eiffel Tower', 'Machine learning'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Use when the user asks about weather, temperature, or conditions in a specific place. "
            "Do NOT use for historical weather or forecasts — current conditions only."
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
            "Use when the user asks about currency conversion or exchange rates. "
            "Use ISO 3-letter codes: USD, EUR, GBP, INR, AUD, SGD, JPY, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "The amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency ISO code, e.g. 'EUR'"},
                "to_currency":   {"type": "string", "description": "Target currency ISO code, e.g. 'USD'"}
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression and return the result. "
            "Use when the user asks for arithmetic, percentages, or any numeric computation. "
            "Input must use numbers and operators only: + - * / ** % ( ). No variables."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '15 * 3 + 7', '2 ** 10', '100 / 4'"
                }
            },
            "required": ["expression"]
        }
    }
]

# ── Tool implementations ───────────────────────────────────────────────────────
# Same as Day 13 — these functions do not handle retries or logging.
# safe_call() is the single place that wraps all of that.

def _first_n_sentences(text: str, n: int = 3) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:n])


def search_wikipedia(query: str) -> dict:
    title = query.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "agentic-ai-learner/1.0"})
    if r.status_code == 404:
        return {"error": f"No Wikipedia page found for '{query}'"}
    r.raise_for_status()
    data = r.json()
    extract = data.get("extract", "").strip()
    if not extract:
        return {"error": "Page exists but has no summary text"}
    return {"title": data.get("title", query), "summary": _first_n_sentences(extract, 3)}


def _geocode_city(city: str) -> Optional[Tuple[float, float]]:
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    return (results[0]["latitude"], results[0]["longitude"]) if results else None


def get_weather(city: str) -> dict:
    coords = _geocode_city(city)
    if coords is None:
        return {"error": f"City not found: '{city}'"}
    lat, lon = coords
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


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    r = requests.get(
        "https://api.frankfurter.app/latest",
        params={"amount": amount, "from": from_currency.upper(), "to": to_currency.upper()},
        timeout=TIMEOUT
    )
    if r.status_code == 422:
        return {"error": "Unknown currency code. Use ISO codes like USD, EUR, GBP."}
    r.raise_for_status()
    rates = r.json().get("rates", {})
    target = to_currency.upper()
    if target not in rates:
        return {"error": f"No rate found for {target}"}
    return {
        "result":        rates[target],
        "from_currency": from_currency.upper(),
        "to_currency":   target,
        "rate":          round(rates[target] / amount, 6),
    }


_SAFE_EXPR = re.compile(r'^[\d\s\+\-\*\/\%\(\)\.\*\*]+$')

def calculate(expression: str) -> dict:
    expr = expression.strip()
    if not _SAFE_EXPR.match(expr):
        return {"error": f"Unsafe expression rejected: '{expr}'. Use numbers and operators only."}
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return {"result": result}
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": f"Could not evaluate: {str(e)}"}


TOOL_REGISTRY = {
    "search_wikipedia": search_wikipedia,
    "get_weather":      get_weather,
    "convert_currency": convert_currency,
    "calculate":        calculate,
}

# ── safe_call ─────────────────────────────────────────────────────────────────
# Single wrapper for every tool dispatch.
# Catches all exception types, returns a typed error dict — never raises.
# Retries once on Timeout only: a timeout might succeed on retry.
# Does NOT retry on HTTPError: a 404 or 500 won't fix itself.

def safe_call(tool_name: str, tool_fn, max_retries: int = 1, **kwargs) -> tuple[dict, float]:
    """
    Call tool_fn(**kwargs) safely.
    Returns (result_dict, duration_ms).
    On Timeout: retries up to max_retries times.
    On any other exception: returns error dict immediately.
    """
    attempt = 0
    start = time.monotonic()

    while True:
        attempt += 1
        try:
            result = tool_fn(**kwargs)
            duration_ms = (time.monotonic() - start) * 1000
            return result, round(duration_ms, 1)

        except requests.Timeout:
            if attempt <= max_retries:
                print(f"     ↻ [{tool_name}] timed out — retrying (attempt {attempt + 1})")
                continue
            duration_ms = (time.monotonic() - start) * 1000
            return {"error": f"Tool timed out after {TIMEOUT}s (retried {max_retries}x)"}, round(duration_ms, 1)

        except requests.HTTPError as e:
            duration_ms = (time.monotonic() - start) * 1000
            return {"error": f"API error {e.response.status_code}"}, round(duration_ms, 1)

        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            return {"error": f"Tool failed: {str(e)}"}, round(duration_ms, 1)

# ── Tool call log ─────────────────────────────────────────────────────────────
# Cleared before each run_agent() call.
# Printed as a summary at the end of each run.

tool_call_log: list[dict] = []


def _log_call(tool: str, input_args: dict, result: dict, duration_ms: float) -> None:
    success = "error" not in result
    tool_call_log.append({
        "tool":        tool,
        "input":       input_args,
        "result":      result,
        "success":     success,
        "duration_ms": duration_ms,
    })


def _print_log() -> None:
    if not tool_call_log:
        return
    print(f"\n{'─'*64}")
    print("  Tool Call Log")
    print(f"{'─'*64}")
    for i, entry in enumerate(tool_call_log, 1):
        status = "✓" if entry["success"] else "✗"
        print(f"  {i}. [{status}] {entry['tool']}  ({entry['duration_ms']}ms)")
        print(f"       input:  {entry['input']}")
        print(f"       result: {entry['result']}")
    passed  = sum(1 for e in tool_call_log if e["success"])
    failed  = len(tool_call_log) - passed
    avg_ms  = round(sum(e["duration_ms"] for e in tool_call_log) / len(tool_call_log), 1)
    print(f"\n  Total: {len(tool_call_log)} calls  |  ✓ {passed} passed  |  ✗ {failed} failed  |  avg {avg_ms}ms")
    print(f"{'─'*64}")

# ── Agent loop ────────────────────────────────────────────────────────────────

def dispatch_all_safe(tool_use_blocks: list) -> list[dict]:
    """
    Run every tool Claude requested, using safe_call for each.
    Checks FORCE_FAIL first — if a tool is in there, skip the real call.
    Returns tool_result messages ready to inject as a user turn.
    """
    results = []
    for block in tool_use_blocks:
        tool_name = block.name
        input_args = block.input

        # Failure injection — swap real call for a forced error
        if tool_name in FORCE_FAIL:
            result     = FORCE_FAIL[tool_name]
            duration_ms = 0.0
            print(f"     ↳ [{tool_name}] ⚡ INJECTED FAILURE  →  {result}")
        else:
            fn = TOOL_REGISTRY.get(tool_name)
            if not fn:
                result, duration_ms = {"error": f"Unknown tool: {tool_name}"}, 0.0
            else:
                result, duration_ms = safe_call(tool_name, fn, **input_args)
            print(f"     ↳ [{tool_name}] ({duration_ms}ms)  →  {result}")

        _log_call(tool_name, input_args, result, duration_ms)

        results.append({
            "type":        "tool_result",
            "tool_use_id": block.id,
            "content":     json.dumps(result),
        })
    return results


def run_agent(question: str) -> str:
    global tool_call_log
    tool_call_log = []  # reset log for each run

    print(f"\n{'='*64}")
    print(f"  Q: {question}")
    if FORCE_FAIL:
        print(f"  ⚡ FORCE_FAIL active: {list(FORCE_FAIL.keys())}")
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

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\n  Answer: {block.text}")
                    _print_log()
                    return block.text
            return "[No text in response]"

        if response.stop_reason == "tool_use":
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            print(f"           tools: {[b.name for b in tool_use_blocks]}")

            messages.append({"role": "assistant", "content": response.content})
            tool_results = dispatch_all_safe(tool_use_blocks)
            messages.append({"role": "user", "content": tool_results})

        else:
            print(f"  [!] Unexpected stop_reason: {response.stop_reason}")
            _print_log()
            return f"Loop ended: {response.stop_reason}"


# ── Tests ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Scenario 1: Normal — all tools working ────────────────────────────────
    print("\n" + "█"*64)
    print("  SCENARIO 1 — Normal operation (no failures)")
    print("█"*64)
    FORCE_FAIL.clear()
    run_agent("What is the weather in Sydney right now?")

    # ── Scenario 2: Normal — multi-tool ───────────────────────────────────────
    print("\n" + "█"*64)
    print("  SCENARIO 2 — Normal operation, two tools")
    print("█"*64)
    FORCE_FAIL.clear()
    run_agent("Who invented Python and what is 2 ** 16?")

    # ── Scenario 3: Injected Wikipedia timeout ────────────────────────────────
    print("\n" + "█"*64)
    print("  SCENARIO 3 — Wikipedia timeout injected")
    print("█"*64)
    FORCE_FAIL["search_wikipedia"] = {"error": "Tool timed out after 5s (retried 1x)"}
    run_agent("Tell me about the history of the internet.")

    # ── Scenario 4: Injected currency API 500 ────────────────────────────────
    print("\n" + "█"*64)
    print("  SCENARIO 4 — Currency API 500 injected")
    print("█"*64)
    FORCE_FAIL.clear()
    FORCE_FAIL["convert_currency"] = {"error": "API error 500"}
    run_agent("How much is 200 EUR in USD?")

    # ── Scenario 5: Two tools fail — agent still gives partial answer ─────────
    print("\n" + "█"*64)
    print("  SCENARIO 5 — Wikipedia + currency both fail, agent gives partial answer")
    print("█"*64)
    FORCE_FAIL["search_wikipedia"] = {"error": "Tool timed out after 5s (retried 1x)"}
    FORCE_FAIL["convert_currency"] = {"error": "API error 500"}
    run_agent("Who invented the Euro and how much is 100 EUR in GBP?")