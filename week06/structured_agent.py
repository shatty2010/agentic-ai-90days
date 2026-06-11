"""
Day 29 — Structured Agent Outputs
Week 6: Agent Architectures | 90-Day Agentic AI Plan

Two approaches to structured output:
  1. JSON instruction in system prompt  → Pydantic validation + retry
  2. Tool-as-output-schema             → submit_answer tool, parse args
"""

import os
import re
import json
import math
import requests
import anthropic
from typing import Literal, Type, TypeVar
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class WeatherReport(BaseModel):
    city:               str
    temp_celsius:       float
    wind_kph:           float
    condition:          str
    packing_suggestion: str


class ResearchAnswer(BaseModel):
    question:    str
    answer:      str
    key_points:  list[str] = Field(min_length=2, max_length=4)
    confidence:  Literal["high", "medium", "low"]


class ComparisonResult(BaseModel):
    item_a:  str
    item_b:  str
    winner:  str
    reason:  str
    metrics: dict[str, str]   # {"metric_name": "value"}


# ---------------------------------------------------------------------------
# Data tools
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


GET_WEATHER_SCHEMA = {
    "name": "get_weather",
    "description": "Get current weather for a city. Call this before returning a WeatherReport.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def _dispatch_get_weather(block) -> dict:
    return get_weather(**block.input)


# ---------------------------------------------------------------------------
# Approach 1 — JSON instruction in system prompt
# ---------------------------------------------------------------------------

def _schema_str(model_cls: Type[BaseModel]) -> str:
    """Return a human-readable JSON schema from a Pydantic model."""
    schema = model_cls.model_json_schema()
    return json.dumps(schema, indent=2)


def structured_call(
    question:       str,
    model_cls:      Type[T],
    extra_tools:    list | None = None,
    max_retries:    int = 3,
) -> T:
    """
    Approach 1: instruct Claude to return JSON matching the Pydantic model schema.
    Validates with model_cls. Retries on ValidationError with the error as feedback.
    """
    schema_text = _schema_str(model_cls)
    system = (
        f"You are a research assistant.\n"
        f"Return your answer as valid JSON matching EXACTLY this schema:\n"
        f"{schema_text}\n\n"
        f"Return ONLY the JSON object. No markdown, no code fences, no explanation."
    )

    tools = extra_tools or []
    messages = [{"role": "user", "content": question}]
    last_error = None

    for attempt in range(1, max_retries + 1):
        if last_error and attempt > 1:
            messages.append({"role": "user", "content": f"Your previous response was invalid:\n{last_error}\nReturn valid JSON."})

        resp = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        # Handle tool calls (e.g. get_weather before returning WeatherReport)
        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use" and block.name == "get_weather":
                    result = _dispatch_get_weather(block)
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
            messages.append({"role": "user", "content": tool_results})
            continue   # loop again to get the JSON response

        # Extract text and validate
        raw = ""
        for block in resp.content:
            if hasattr(block, "text"):
                raw = block.text.strip()
                break

        # Strip markdown code fences the model occasionally wraps around JSON
        raw = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip()

        try:
            return model_cls.model_validate_json(raw)
        except (ValidationError, Exception) as e:
            last_error = str(e)
            print(f"  [Attempt {attempt}] Validation failed: {last_error[:120]}")

    raise ValueError(f"Structured call failed after {max_retries} attempts. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Approach 2 — Tool-as-output-schema
# ---------------------------------------------------------------------------

def _make_submit_tool(model_cls: Type[BaseModel]) -> dict:
    """Build a 'submit_answer' tool whose input_schema is the Pydantic model's JSON schema."""
    schema = model_cls.model_json_schema()
    return {
        "name": "submit_answer",
        "description": f"Submit your final structured answer as a {model_cls.__name__}.",
        "input_schema": {
            "type":       schema.get("type", "object"),
            "properties": schema.get("properties", {}),
            "required":   schema.get("required", []),
        },
    }


def structured_tool_call(
    question:    str,
    model_cls:   Type[T],
    data_tools:  list | None = None,
    max_retries: int = 3,
) -> T:
    """
    Approach 2: Claude MUST call submit_answer to return its answer.
    Tool args are the structured output — parsed and validated by Pydantic.
    """
    submit_tool  = _make_submit_tool(model_cls)
    all_tools    = (data_tools or []) + [submit_tool]
    tool_choice  = {"type": "any"}   # force Claude to use a tool

    system = (
        "You are a research assistant. "
        "When you have your final answer, you MUST call the submit_answer tool. "
        "Do not return free text as your final response."
    )
    messages    = [{"role": "user", "content": question}]
    last_error  = None

    for attempt in range(1, max_retries + 1):
        if last_error and attempt > 1:
            messages.append({"role": "user", "content": f"Your previous submit_answer call was invalid:\n{last_error}\nCall submit_answer again with correct arguments."})

        resp = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system,
            tools=all_tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        # Dispatch data tools or capture submit_answer
        data_results = []
        submit_args  = None

        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "submit_answer":
                submit_args = block.input
                messages.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps({"status": "received"}),
                }]})
            elif block.name == "get_weather":
                result = _dispatch_get_weather(block)
                data_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        if data_results:
            messages.append({"role": "user", "content": data_results})

        if submit_args is not None:
            try:
                return model_cls.model_validate(submit_args)
            except ValidationError as e:
                last_error = str(e)
                print(f"  [Attempt {attempt}] submit_answer validation failed: {last_error[:120]}")

    raise ValueError(f"Structured tool call failed after {max_retries} attempts.")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "="*64)
    print("TEST 1 — WeatherReport (tool call + Approach 1)")
    print("="*64)
    report: WeatherReport = structured_call(
        question="What is the current weather in London? Provide a packing suggestion.",
        model_cls=WeatherReport,
        extra_tools=[GET_WEATHER_SCHEMA],
    )
    print(f"  Parsed: {report.model_dump_json(indent=2)}")

    print("\n" + "="*64)
    print("TEST 2 — ResearchAnswer (Approach 1 — no tools)")
    print("="*64)
    research: ResearchAnswer = structured_call(
        question="What is the ReAct pattern in AI agents?",
        model_cls=ResearchAnswer,
    )
    print(f"  Parsed: {research.model_dump_json(indent=2)}")

    print("\n" + "="*64)
    print("TEST 3 — ComparisonResult (Approach 2 — tool-as-schema)")
    print("="*64)
    comparison: ComparisonResult = structured_tool_call(
        question="Compare RAG vs fine-tuning for knowledge injection in LLMs.",
        model_cls=ComparisonResult,
    )
    print(f"  Parsed: {comparison.model_dump_json(indent=2)}")
