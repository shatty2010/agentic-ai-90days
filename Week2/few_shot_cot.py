import time
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

TEST_INPUTS = [
    "Works great, very fast delivery.",
    "Stopped working after one week. Terrible.",
    "Good features but the battery is awful.",
    "Exactly as described. Happy with it.",
    "Screen is beautiful but the software crashes constantly.",
]

ZERO_SHOT_SYSTEM = (
    "You are a sentiment classifier. "
    "Reply with only one word: positive, negative, or mixed."
)

FEW_SHOT_SYSTEM = """You are a sentiment classifier. Reply with only one word: positive, negative, or mixed.

Examples:
"Works great, very fast." → positive
"Stopped working after a week." → negative
"Good features but terrible battery." → mixed"""

COT_SYSTEM = """You are a sentiment classifier.

Think step by step before answering:
1. List the positive aspects mentioned
2. List the negative aspects mentioned
3. Classify as positive, negative, or mixed

Respond in this format:
Positives: <list or 'none'>
Negatives: <list or 'none'>
Classification: <positive|negative|mixed>"""


def run_strategy(strategy_name: str, system: str, inputs: list) -> list:
    results = []
    for text in inputs:
        start = time.time()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=system,
            messages=[{"role": "user", "content": f"Classify: {text}"}]
        )
        latency = round((time.time() - start) * 1000)
        block = msg.content[0]
        if isinstance(block, anthropic.types.TextBlock):
            response_text = block.text.strip().replace("\n", " | ")
        else:
            response_text = str(block)
        results.append({
            "strategy": strategy_name,
            "input": text[:40],
            "output": response_text,
            "tokens": msg.usage.input_tokens + msg.usage.output_tokens,
            "latency_ms": latency,
        })
    return results


def print_table(results: list):
    print(f"\n{'Strategy':<12} {'Input':<42} {'Output':<50} {'Tokens':>7} {'ms':>6}")
    print("-" * 125)
    for r in results:
        print(f"{r['strategy']:<12} {r['input']:<42} {r['output']:<50} {r['tokens']:>7} {r['latency_ms']:>6}")


if __name__ == "__main__":
    all_results = []
    for name, system in [
        ("zero-shot", ZERO_SHOT_SYSTEM),
        ("few-shot",  FEW_SHOT_SYSTEM),
        ("cot",       COT_SYSTEM),
    ]:
        print(f"Running {name}...")
        all_results.extend(run_strategy(name, system, TEST_INPUTS))

    print_table(all_results)

    # Token cost summary
    for name in ["zero-shot", "few-shot", "cot"]:
        total = sum(r["tokens"] for r in all_results if r["strategy"] == name)
        print(f"{name}: {total} total tokens across 5 inputs")