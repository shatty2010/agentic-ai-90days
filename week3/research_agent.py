"""
Day 15 — Week 3 Capstone: Research Agent with Real APIs
---------------------------------------------------------
Combines everything from Days 11–14 into a single production-quality agent.

Components:
  • ConversationManager   — sliding window memory (Day 8 pattern, included inline)
  • 4 real API tools      — Wikipedia, Open-Meteo weather, Frankfurter currency, calculator
  • safe_call             — catches all exceptions, retries once on Timeout (Day 14)
  • dispatch_tools        — handles single and parallel tool_use blocks (Day 13)
  • run_agent             — full loop with max_steps guard and tool call logging
  • tool_call_log         — per-run log of every tool call with timing and success flag
"""

import json
import os
import re
import time
from collections import deque
from typing import Optional, Tuple

import anthropic
from google import genai
from google.genai import types
import requests
from dotenv import load_dotenv

load_dotenv()

# Instantiating Anthropic and modern Google GenAI Clients
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = genai.Client()  # Automatically picks up GEMINI_API_KEY from environment

GEMINI_MODEL = "gemini-2.5-flash"
TIMEOUT    = 5    # HTTP timeout in seconds
MAX_WINDOW = 20   # ConversationManager: max messages kept in sliding window

# ── ConversationManager ───────────────────────────────────────────────────────
class ConversationManager:
    """
    Maintains a bounded message history for multi-turn conversations.
    """
    def __init__(self, system_prompt: str = "", max_messages: int = MAX_WINDOW):
        self.system_prompt = system_prompt
        self.max_messages  = max_messages
        self._history: deque = deque(maxlen=max_messages)

    def add(self, role: str, content) -> None:
        self._history.append({"role": role, "content": content})

    def messages(self) -> list:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()

# ── WMO weather codes ─────────────────────────────────────────────────────────
WMO_CODES: dict[int, str] = {
    0:  "Clear sky",      1:  "Mainly clear",    2:  "Partly cloudy",
    3:  "Overcast",       45: "Fog",              48: "Icy fog",
    51: "Light drizzle",  53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain",    63: "Moderate rain",    65: "Heavy rain",
    71: "Slight snow",    73: "Moderate snow",    75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Heavy showers",
    95: "Thunderstorm",   96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}

# ── Tool schemas ───────────────────────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "name": "search_wikipedia",
        "description": (
            "Search Wikipedia for factual background on a topic. "
            "Use when the question asks about historical facts, definitions, people, "
            "places, events, or scientific concepts. "
            "Also use to identify the capital city of a country. "
            "Do NOT use for current prices, live weather, or real-time data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic to look up. Use the most likely Wikipedia article title."
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
            "in a specific place right now."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Paris', 'Tokyo', 'Sydney'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "convert_currency",
        "description": (
            "Convert an amount from one currency to another using live exchange rates. "
            "Use standard ISO 3-letter codes: USD, EUR, GBP, AUD, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "The numeric amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency ISO code, e.g. 'EUR'"},
                "to_currency": {"type": "string", "description": "Target currency ISO code, e.g. 'USD'"}
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression and return the numeric result. "
            "Input must use numbers and operators only: + - * / ** % ( )."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A safe math expression"}
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
    r = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
        timeout=TIMEOUT,
        headers={"User-Agent": "agentic-ai-learner/1.0"}
    )
    if r.status_code == 404:
        return {"error": f"No Wikipedia page found for '{query}'"}
    r.raise_for_status()
    data = r.json()
    extract = data.get("extract", "").strip()
    if not extract:
        return {"error": "Page exists but has no summary text"}
    return {
        "title":   data.get("title", query),
        "summary": _first_n_sentences(extract, 3),
    }

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
        params={
            "amount": amount,
            "from":   from_currency.upper(),
            "to":     to_currency.upper(),
        },
        timeout=TIMEOUT
    )
    if r.status_code == 422:
        return {"error": "Unknown currency code. Use ISO codes: USD, EUR, GBP, AUD..."}
    r.raise_for_status()
    rates  = r.json().get("rates", {})
    target = to_currency.upper()
    if target not in rates:
        return {"error": f"No rate found for {target}"}
    return {
        "result":      rates[target],
        "from_currency": from_currency.upper(),
        "to_currency":   target,
        "rate":          round(rates[target] / amount, 6),
    }

_SAFE_EXPR = re.compile(r'^[\d\s\+\-\*\/\%\(\)\.\*\*]+$')

def calculate(expression: str) -> dict:
    expr = expression.strip()
    if not _SAFE_EXPR.match(expr):
        return {"error": f"Unsafe expression: '{expr}'. Use numbers and operators only."}
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
def safe_call(tool_name: str, tool_fn, max_retries: int = 1, **kwargs) -> tuple[dict, float]:
    attempt = 0
    start   = time.monotonic()
    while True:
        attempt += 1
        try:
            result      = tool_fn(**kwargs)
            duration_ms = (time.monotonic() - start) * 1000
            return result, round(duration_ms, 1)
        except requests.Timeout:
            if attempt <= max_retries:
                print(f"     ↻ [{tool_name}] timed out — retrying ({attempt + 1})")
                continue
            duration_ms = (time.monotonic() - start) * 1000
            return {"error": f"Timed out after {TIMEOUT}s (retried {max_retries}x)"}, round(duration_ms, 1)
        except requests.HTTPError as e:
            duration_ms = (time.monotonic() - start) * 1000
            return {"error": f"API error {e.response.status_code}"}, round(duration_ms, 1)
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            return {"error": f"Tool failed: {str(e)}"}, round(duration_ms, 1)

# ── Tool call log ─────────────────────────────────────────────────────────────
tool_call_log: list[dict] = []

def _log(tool: str, input_args: dict, result: dict, duration_ms: float) -> None:
    tool_call_log.append({
        "tool":        tool,
        "input":       input_args,
        "result":      result,
        "success":     "error" not in result,
        "duration_ms": duration_ms,
    })

def _print_log() -> None:
    if not tool_call_log:
        return
    print(f"\n{'─'*64}")
    print("  Tool Call Log")
    print(f"{'─'*64}")
    for i, e in enumerate(tool_call_log, 1):
        icon = "✓" if e["success"] else "✗"
        print(f"  {i}. [{icon}] {e['tool']}  ({e['duration_ms']}ms)")
        print(f"       input:  {e['input']}")
        print(f"       result: {e['result']}")
    passed = sum(1 for e in tool_call_log if e["success"])
    failed = len(tool_call_log) - passed
    avg_ms = round(sum(e["duration_ms"] for e in tool_call_log) / len(tool_call_log), 1)
    print(f"\n  {len(tool_call_log)} calls  |  ✓ {passed}  |  ✗ {failed}  |  avg {avg_ms}ms")
    print(f"{'─'*64}\n")

# ── dispatch_tools (Anthropic Variant) ──────────────────────────────────────────
def dispatch_tools(tool_use_blocks: list) -> list[dict]:
    results = []
    for block in tool_use_blocks:
        fn = TOOL_REGISTRY.get(block.name)
        if not fn:
            result, duration_ms = {"error": f"Unknown tool: {block.name}"}, 0.0
        else:
            result, duration_ms = safe_call(block.name, fn, **block.input)

        print(f"     ↳ [{block.name}] ({duration_ms}ms)  input={block.input}")
        print(f"       result: {result}")
        _log(block.name, block.input, result, duration_ms)

        results.append({
            "type":        "tool_result",
            "tool_use_id": block.id,
            "content":     json.dumps(result),
        })
    return results

# ── Gemini helpers ────────────────────────────────────────────────────────────
def _build_gemini_tools() -> list:
    declarations = []
    for s in TOOL_SCHEMAS:
        # Map parameters cleanly converting 'input_schema' architecture
        props = s["input_schema"].get("properties", {})
        req = s["input_schema"].get("required", [])
        
        declarations.append(
            types.FunctionDeclaration(
                name=s["name"],
                description=s["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={k: types.Schema(type=v["type"].upper(), description=v.get("description", "")) for k, v in props.items()},
                    required=req
                )
            )
        )
    return [types.Tool(function_declarations=declarations)]

def _run_gemini_agent(question: str, max_steps: int = 10) -> str:
    global tool_call_log
    tool_call_log = []

    chat = gemini_client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=_build_gemini_tools(),
        )
    )

    print(f"\n{'='*64}")
    print(f"  Q: {question}  [Gemini]")
    print(f"{'='*64}")

    current_msg = question
    for step in range(1, max_steps + 1):
        print(f"\n[Step {step}] Calling Gemini...")
        try:
            response = chat.send_message(current_msg)
        except Exception as e:
            print(f"  [!] Gemini API error: {e}")
            _print_log()
            return f"Gemini API error: {e}"

        if not response.function_calls:
            print("           stop_reason: end_turn")
            answer = response.text
            print(f"\n  Answer: {answer}")
            _print_log()
            return answer

        print("           stop_reason: tool_use")
        print(f"           tools: {[fc.name for fc in response.function_calls]}")

        # Execute parallel or sequential calls generated by the step loop
        parts = []
        for fc in response.function_calls:
            fn_args: dict = dict(fc.args) if fc.args else {}
            fn = TOOL_REGISTRY.get(fc.name)
            
            if fn:
                result, ms = safe_call(fc.name, fn, **fn_args)
            else:
                result, ms = {"error": f"Unknown tool: {fc.name}"}, 0.0

            print(f"     ↳ [{fc.name}] ({ms}ms)  input={fn_args}")
            print(f"       result: {result}")
            _log(fc.name, fn_args, result, ms)

            # Append the structured Function Response Part
            parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result}
                )
            )
        
        # Prepare parts array payload for the next loop turn execution
        current_msg = parts

    _print_log()
    return f"[Agent stopped: reached max_steps={max_steps}]"

# ── run_agent ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a research assistant with access to real-time tools. "
    "When answering questions that require current data — weather, exchange rates, "
    "facts — always use the appropriate tool rather than guessing. "
    "If a tool returns an error, tell the user clearly and answer what you can "
    "from your own knowledge."
)

def run_agent(question: str, max_steps: int = 10, provider: str = "anthropic") -> str:
    if provider == "gemini":
        return _run_gemini_agent(question, max_steps)

    global tool_call_log
    tool_call_log = []

    conversation = ConversationManager(system_prompt=SYSTEM_PROMPT)
    conversation.add("user", question)

    print(f"\n{'='*64}")
    print(f"  Q: {question} [Anthropic]")
    print(f"{'='*64}")

    for step in range(1, max_steps + 1):
        print(f"\n[Step {step}] Calling Claude...")

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=conversation.messages(),
            )
        except anthropic.APIError as e:
            print(f"  [!] Anthropic API error: {e}")
            _print_log()
            return f"Anthropic API error: {e}"

        print(f"           stop_reason: {response.stop_reason}")

        if response.stop_reason == "end_turn":
            answer = next(
                (b.text for b in response.content if hasattr(b, "text")),
                "[No text in response]"
            )
            print(f"\n  Answer: {answer}")
            _print_log()
            return answer

        if response.stop_reason == "tool_use":
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            print(f"           tools: {[b.name for b in tool_use_blocks]}")

            conversation.add("assistant", response.content)
            tool_results = dispatch_tools(tool_use_blocks)
            conversation.add("user", tool_results)
        else:
            print(f"  [!] Unexpected stop_reason: {response.stop_reason}")
            _print_log()
            return f"Loop ended unexpectedly: {response.stop_reason}"

    _print_log()
    return f"[Agent stopped: reached max_steps={max_steps} without a final answer]"

# ── Tests ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROOF_Q  = ("What is the current weather in the capital of France, "
                "what is its population, and how much is €100 in USD right now?")
    PARALLEL_Q = "Who invented Python the programming language, and what is 2 ** 32?"
    CALC_Q     = "What is 15% of 3847?"

    run_agent(PROOF_Q)
    run_agent(PARALLEL_Q)
    run_agent(CALC_Q)

    # Run Gemini execution test blocks cleanly
    run_agent(PROOF_Q,     provider="gemini")
    run_agent(PARALLEL_Q,  provider="gemini")
    run_agent(CALC_Q,      provider="gemini")