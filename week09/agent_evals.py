"""
Day 42 — Evaluations for Agentic Systems
Week 9: Production, Security & Observability | 90-Day Agentic AI Plan

An eval harness: exact-match, rubric-keyword, and LLM-as-judge scoring,
run multiple times to surface flaky cases rather than trusting one average.

Requires:
  pip install anthropic
  Optional: pip install azure-ai-evaluation (for the managed-evaluator demo)
"""

import os
import sys
import statistics
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"


def call_llm(prompt: str, max_tokens: int = 300) -> str:
    resp = _client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# The system under test — a stand-in for the Week 8/9 pipeline
# ---------------------------------------------------------------------------

def run_pipeline(question: str) -> str:
    return call_llm(
        f"Answer concisely (2-3 sentences), as a cloud architecture assistant: {question}"
    )


# ---------------------------------------------------------------------------
# Scoring functions — increasing cost, decreasing brittleness
# ---------------------------------------------------------------------------

def score_exact_match(actual: str, expected: str) -> bool:
    return actual.strip().lower() == expected.strip().lower()


def score_rubric(actual: str, must_contain: list) -> float:
    hits = sum(1 for kw in must_contain if kw.lower() in actual.lower())
    return round(hits / len(must_contain), 2) if must_contain else 1.0


JUDGE_PROMPT = """You are grading an AI agent's response for correctness and completeness.

Question: {question}
Agent's response: {response}
Grading criteria: {criteria}

Score 1 if the response satisfies the criteria, 0 if it does not.
Respond with only the number, nothing else."""


def score_llm_judge(question: str, response: str, criteria: str) -> int:
    raw = call_llm(JUDGE_PROMPT.format(question=question, response=response, criteria=criteria))
    digits = "".join(c for c in raw if c in "01")
    return int(digits[0]) if digits else 0


def validate_judge() -> bool:
    """Sanity-check the judge against a hand-graded example before trusting it."""
    good = score_llm_judge(
        "What is RAG?",
        "RAG retrieves relevant documents and injects them into the prompt before generation.",
        "Mentions retrieval and injecting context into the prompt",
    )
    bad = score_llm_judge(
        "What is RAG?",
        "RAG is a type of rice dish popular in some cuisines.",
        "Mentions retrieval and injecting context into the prompt",
    )
    ok = (good == 1) and (bad == 0)
    print(f"  [Judge validation] good example scored {good} (expect 1), "
          f"bad example scored {bad} (expect 0) — {'OK' if ok else 'JUDGE UNRELIABLE, do not trust it yet'}")
    return ok


# ---------------------------------------------------------------------------
# Eval dataset
# ---------------------------------------------------------------------------

EVAL_DATASET = [
    {
        "question": "What's the main architectural difference between Bedrock Agents and Azure AI Agent Service for tool calling?",
        "criteria": "Correctly states that Bedrock routes tool calls to a Lambda function while Azure executes the tool function in the caller's own process.",
        "must_contain": ["lambda", "process"],
    },
    {
        "question": "Why does a bank need a VPC endpoint for Bedrock rather than calling the public API?",
        "criteria": "Explains that a VPC endpoint keeps traffic on AWS's private network instead of transiting the public internet.",
        "must_contain": ["private", "internet"],
    },
    {
        "question": "What does DefaultAzureCredential try first when running inside an Azure Container App?",
        "criteria": "Correctly identifies managed identity as the credential used in Azure-hosted compute, ahead of CLI login.",
        "must_contain": ["managed identity"],
    },
    {
        "question": "In one word, what AWS service provides an audit trail of API calls?",
        "expected_tool": "CloudTrail",
        "criteria": "Names CloudTrail specifically.",
        "must_contain": ["cloudtrail"],
    },
    {
        "question": "What's the purpose of tokenizing PII before sending a prompt to an LLM, even over a private network?",
        "criteria": "Explains that network isolation protects data in transit but not from the model itself seeing raw PII, so tokenization is a separate, additional control.",
        "must_contain": ["model", "raw"],
    },
]


def run_eval_suite(cases: list, n_runs: int = 3) -> dict:
    per_case_scores = {c["question"]: [] for c in cases}

    for run in range(1, n_runs + 1):
        print(f"\n  --- Run {run}/{n_runs} ---")
        for case in cases:
            response = run_pipeline(case["question"])
            if case.get("expected_tool"):
                score = 1 if score_exact_match(response, case["expected_tool"]) else score_llm_judge(
                    case["question"], response, case["criteria"]
                )
            else:
                score = score_llm_judge(case["question"], response, case["criteria"])
            per_case_scores[case["question"]].append(score)
            print(f"    [{score}] {case['question'][:60]}...")

    print("\n  --- Summary ---")
    print(f"  {'case':<62} {'runs':<12} {'avg':<6} flaky?")
    overall = []
    for question, scores in per_case_scores.items():
        avg = statistics.mean(scores)
        is_flaky = len(set(scores)) > 1
        overall.extend(scores)
        print(f"  {question[:60]:<62} {str(scores):<12} {avg:<6.2f} {'YES' if is_flaky else 'no'}")

    pass_rate = statistics.mean(overall)
    return {"pass_rate": round(pass_rate, 3), "n_cases": len(cases), "n_runs": n_runs}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent eval harness / CI gate")
    parser.add_argument("--gate", action="store_true",
                         help="Exit nonzero if pass_rate falls below --threshold (for CI)")
    parser.add_argument("--threshold", type=float, default=0.8,
                         help="Minimum pass rate required to pass the gate")
    parser.add_argument("--runs", type=int, default=3, help="Runs per case, to surface flakiness")
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("STEP 1 — Validate the judge before trusting it")
    print("=" * 64)
    judge_ok = validate_judge()
    if not judge_ok and args.gate:
        print("  [GATE FAILED] Judge failed its own sanity check — refusing to trust its scores.")
        sys.exit(1)

    print("\n" + "=" * 64)
    print(f"STEP 2 — Run the eval suite ({args.runs}x per case, to surface flakiness)")
    print("=" * 64)
    summary = run_eval_suite(EVAL_DATASET, n_runs=args.runs)

    print("\n" + "=" * 64)
    print(f"OVERALL PASS RATE: {summary['pass_rate']*100:.1f}% "
          f"across {summary['n_cases']} cases x {summary['n_runs']} runs")
    print("=" * 64)

    if summary["pass_rate"] < args.threshold:
        print(f"  [GATE] Below {args.threshold*100:.0f}% threshold — would block deploy in a CI gate.")
        if args.gate:
            sys.exit(1)   # nonzero exit is what actually fails the CI job
    else:
        print(f"  [GATE] Above {args.threshold*100:.0f}% threshold — would pass a CI gate.")
