"""
Day 44 — Cost & Latency Optimization
Week 9: Production, Security & Observability | 90-Day Agentic AI Plan

Complexity-based model routing, prompt caching, and per-request cost tracking.

Requires:
  pip install anthropic
"""

import re
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Complexity-based routing
# ---------------------------------------------------------------------------

MODEL_TIERS = {
    "simple":  "claude-haiku-4-5-20251001",
    "medium":  "claude-haiku-4-5-20251001",
    "complex": "claude-opus-4-8",
}


def estimate_complexity(task: str) -> str:
    word_count = len(task.split())
    has_multi_step_language = bool(re.search(
        r"\b(compare|analyze|analyse|synthesize|synthesise|recommend|tradeoff|why|explain)\b",
        task.lower(),
    ))
    if word_count < 15 and not has_multi_step_language:
        return "simple"
    if has_multi_step_language or word_count > 40:
        return "complex"
    return "medium"


def route(task: str) -> str:
    tier = estimate_complexity(task)
    return MODEL_TIERS[tier]


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------

class CachedPromptClient:

    def chat(self, system_stable: str, user_variable: str, model: str) -> tuple:
        resp = _client.messages.create(
            model=model,
            max_tokens=400,
            system=[{
                "type": "text",
                "text": system_stable,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_variable}],
        )
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        }
        return resp.content[0].text.strip(), usage


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

PRICING_PER_MILLION_TOKENS = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-opus-4-8":           {"input": 15.00, "output": 75.00},
}


def track_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_PER_MILLION_TOKENS[model]
    cost = (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]
    return round(cost, 6)


class CostTracker:
    def __init__(self):
        self.calls = []

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": track_cost(model, input_tokens, output_tokens),
        })

    def summary(self) -> dict:
        total = sum(c["cost"] for c in self.calls)
        by_model = {}
        for c in self.calls:
            by_model.setdefault(c["model"], 0.0)
            by_model[c["model"]] += c["cost"]

        # counterfactual: what if every call had gone to the most expensive model instead
        most_expensive = max(PRICING_PER_MILLION_TOKENS, key=lambda m: PRICING_PER_MILLION_TOKENS[m]["output"])
        counterfactual = sum(
            track_cost(most_expensive, c["input_tokens"], c["output_tokens"]) for c in self.calls
        )
        savings_pct = round((1 - total / counterfactual) * 100, 1) if counterfactual else 0.0

        return {
            "total_cost": round(total, 6),
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
            "call_count": len(self.calls),
            "counterfactual_if_all_opus": round(counterfactual, 6),
            "savings_pct": savings_pct,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TEST_TASKS = [
    "What tool should handle a weather lookup?",
    "Extract the date from: the meeting is on July 9th, 2026.",
    "Summarize this in one sentence: Bedrock and Azure both offer managed agent services.",
    "Compare Bedrock Agents and Azure AI Agent Service and explain the tradeoff for a startup with a small team.",
    "Why would a bank choose VPC-isolated Bedrock over a public API, and what's the compliance tradeoff versus development speed?",
]

STABLE_SYSTEM_PROMPT = (
    "You are a cloud architecture assistant specializing in AWS and Azure agentic AI services. "
    "Answer concisely, in 2-3 sentences, grounded in the actual service capabilities of Bedrock "
    "and Azure AI Foundry. Do not speculate about unannounced features."
)

if __name__ == "__main__":
    tracker = CostTracker()
    cached_client = CachedPromptClient()

    print("\n" + "=" * 64)
    print("STEP 1 — Route each task to a model tier by complexity")
    print("=" * 64)
    for task in TEST_TASKS:
        tier = estimate_complexity(task)
        model = MODEL_TIERS[tier]
        print(f"  [{tier:<7}] -> {model:<28} | {task[:60]}")

    print("\n" + "=" * 64)
    print("STEP 2 — Run with prompt caching (same system prompt across calls)")
    print("=" * 64)
    for i, task in enumerate(TEST_TASKS):
        model = route(task)
        answer, usage = cached_client.chat(STABLE_SYSTEM_PROMPT, task, model)
        tracker.record(model, usage["input_tokens"], usage["output_tokens"])
        cache_note = (f"cache_read={usage['cache_read_input_tokens']}, "
                      f"cache_created={usage['cache_creation_input_tokens']}")
        print(f"\n  Call {i+1} [{model}] ({cache_note})")
        print(f"    Q: {task[:70]}")
        print(f"    A: {answer[:150]}")

    print("\n" + "=" * 64)
    print("STEP 3 — Cost summary")
    print("=" * 64)
    summary = tracker.summary()
    print(f"  Total cost (actual routing):      ${summary['total_cost']:.6f}")
    print(f"  By model: {summary['by_model']}")
    print(f"  Counterfactual (all Opus):         ${summary['counterfactual_if_all_opus']:.6f}")
    print(f"  Savings from complexity routing:   {summary['savings_pct']}%")
