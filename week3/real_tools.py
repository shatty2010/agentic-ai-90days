"""
Day 12 — Real External APIs as Tools
-------------------------------------
Goal: Replace hardcoded tool responses with real HTTP calls.

APIs used (both free, no API key required):
  • Wikipedia REST API  →  /api/rest_v1/page/summary/{title}
  • Open-Meteo          →  geocoding + current weather

Agent loop is the same pattern as Day 11.
What changes: tool functions make real HTTP requests, return cleaned dicts,
and handle every failure mode without raising.

Run:
    pip install anthropic requests python-dotenv
    export ANTHROPIC_API_KEY=your_key
    python real_tools.py
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

TIMEOUT = 5  # seconds — applied to every HTTP call

# ── WMO weather code → human-readable string ─────────────────────────────────
# Open-Meteo returns an integer weather code (WMO standard), not a string.
# This map is the only way to make it readable.

WMO_CODES: dict[int, str] = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Fog",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Heavy showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}

# ── Tool schemas ──────────────────────────────────────────────────────────────
# The description field is the most important thing here.
# It tells Claude *when* to use this tool — not just what it does.
# Vague descriptions → wrong tool selection. Specific ones → reliable selection.

TOOL_SCHEMAS = [
    {
        "name": "search_wikipedia",
        "description": (
            "Search Wikipedia for factual background on a topic. "
            "Use when the question asks about historical facts, definitions, people, "
            "places, events, or concepts. "
            "Do NOT use for current events, live data, or real-time information — "
            "Wikipedia content is not up to the minute."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The topic to search for. Use the most likely Wikipedia article title, "
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
            "Use when the user asks about weather, temperature, or conditions in a specific place. "
            "Returns current conditions only — not historical data or forecasts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city, e.g. 'London', 'Mumbai', 'Singapore'"
                }
            },
            "required": ["city"]
        }
    }
]

# ── Tool implementations ───────────────────────────────────────────────────────
# Design rule: never pass a raw API response to Claude.
# Extract only the 2–3 fields that matter and return a clean dict.
# Think of each function as a DTO mapper — raw API row goes in, clean struct comes out.
# On any failure: return {'error': reason} — never raise. The agent loop must keep running.

def _first_n_sentences(text: str, n: int = 3) -> str:
    """Return the first n sentences from a text block."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:n])


def search_wikipedia(query: str) -> dict:
    """
    Fetch the Wikipedia page summary for a topic.
    Returns the page title and first 3 sentences of the extract.

    API: GET https://en.wikipedia.org/api/rest_v1/page/summary/{title}
    No API key needed.
    """
    # Wikipedia titles use underscores; spaces cause a redirect but underscores are cleaner
    title = query.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    try:
        r = requests.get(
            url,
            timeout=TIMEOUT,
            headers={"User-Agent": "agentic-ai-learner/1.0 (day12-exercise)"}
        )

        if r.status_code == 404:
            return {"error": f"No Wikipedia page found for '{query}'"}

        r.raise_for_status()  # catch any other 4xx / 5xx

        data = r.json()
        extract = data.get("extract", "").strip()

        if not extract:
            return {"error": "Page exists but has no summary text"}

        return {
            "title":   data.get("title", query),
            "summary": _first_n_sentences(extract, 3)
        }

    except requests.Timeout:
        return {"error": "Wikipedia request timed out after 5s"}
    except requests.HTTPError as e:
        return {"error": f"Wikipedia HTTP error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Wikipedia lookup failed: {str(e)}"}


def _geocode_city(city: str) -> Optional[Tuple[float, float]]:
    """
    Convert a city name to (latitude, longitude) using Open-Meteo's geocoding API.
    Returns None if the city is not found or the request fails.

    API: GET https://geocoding-api.open-meteo.com/v1/search
    No API key needed.
    """
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=TIMEOUT
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        return results[0]["latitude"], results[0]["longitude"]
    except Exception:
        return None


def get_weather(city: str) -> dict:
    """
    Get current weather for a city.
    Step 1: geocode the city to (lat, lon).
    Step 2: fetch current_weather from Open-Meteo forecast API.

    API: GET https://api.open-meteo.com/v1/forecast
    No API key needed.
    """
    coords = _geocode_city(city)
    if coords is None:
        return {"error": f"City not found or geocoding failed: '{city}'"}

    lat, lon = coords

    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":        lat,
                "longitude":       lon,
                "current_weather": True,
            },
            timeout=TIMEOUT
        )
        r.raise_for_status()

        cw = r.json().get("current_weather")
        if not cw:
            return {"error": "Weather data unavailable for this location"}

        condition = WMO_CODES.get(cw.get("weathercode", -1), "Unknown")

        return {
            "temp_celsius": cw["temperature"],
            "condition":    condition,
        }

    except requests.Timeout:
        return {"error": "Weather API request timed out after 5s"}
    except requests.HTTPError as e:
        return {"error": f"Weather API error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Weather lookup failed: {str(e)}"}


TOOL_REGISTRY = {
    "search_wikipedia": search_wikipedia,
    "get_weather":      get_weather,
}

# ── Agent loop ────────────────────────────────────────────────────────────────
# Same pattern as Day 11.
# The only thing that changed is what the tool functions do internally.

def dispatch_tool(tool_use_block) -> dict:
    """Run the tool Claude requested. Returns a dict, always."""
    name = tool_use_block.name
    args = tool_use_block.input
    fn   = TOOL_REGISTRY.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    return fn(**args)


def run_agent(question: str) -> str:
    """
    Tool-use agent loop.

    Messages flow:
      user       → question
      assistant  → [tool_use block]      (Claude picked a tool)
      user       → [tool_result block]   (we ran the tool and returned the result)
      assistant  → final text answer     (stop_reason == 'end_turn')
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
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        print(f"           stop_reason: {response.stop_reason}")

        # ── Done — return the text answer ─────────────────────────────────────
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\n  Answer: {block.text}")
                    return block.text
            return "[No text in response]"

        # ── Tool call requested ───────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            # Append Claude's response (contains the tool_use block)
            messages.append({"role": "assistant", "content": response.content})

            # Dispatch every tool_use block — Claude can request more than one per turn
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"           tool call: {block.name}({block.input})")
                    result = dispatch_tool(block)
                    print(f"           result:    {result}")
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,   # must match Claude's request ID exactly
                        "content":     json.dumps(result),
                    })

            # Inject all results in one user turn — not separate turns
            messages.append({"role": "user", "content": tool_results})

            # Loop — Claude reads the results and responds

        else:
            print(f"  [!] Unexpected stop_reason: {response.stop_reason}")
            return f"Loop ended: {response.stop_reason}"


# ── Tests ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        # 3 Wikipedia queries
        "What is machine learning?",
        "Tell me about the Eiffel Tower.",
        "Who was Ada Lovelace?",

        # 2 weather queries
        "What is the current weather in Sydney?",
        "Is it cold in Berlin right now?",

        # Deliberate bad query — tests error path end-to-end
        "What is the weather in XYZABC123NotACity?",
    ]

    for q in test_questions:
        run_agent(q)