import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ── Fake tool registry ─────────────────────────────────────────
FAKE_TOOLS = {
    "get_weather": {
        "london":  "Cloudy, 15°C",
        "tokyo":   "Sunny, 28°C",
        "sydney":  "Rainy, 18°C",
    },
    "get_time": {
        "europe/london":  "10:15 AM GMT",
        "asia/tokyo":     "3:42 PM JST",
        "australia/sydney":"9:15 PM AEDT",
    },
}

def fake_tool_call(tool_name: str, tool_input: str) -> str:
    tool_input = tool_input.strip().strip('"\'').lower()
    result = FAKE_TOOLS.get(tool_name, {}).get(tool_input)
    return result if result else f"No data found for '{tool_input}'"

# ── ReAct system prompt ────────────────────────────────────────
def build_system_prompt() -> str:
    return """You are a helpful assistant that solves problems step by step.

Available tools:
- get_weather(city: str) → returns current weather
- get_time(timezone: str) → returns current local time

Rules:
1. Always start with a Thought
2. Use exactly one Action per turn
3. Wait for the Observation before your next Thought
4. Use Answer: only when you have all the information needed

Format:
Thought: <reasoning>
Action: <tool_name>("<input>")

Or when done:
Answer: <final answer>"""

# ── Parse action from model output ────────────────────────────
import re

def parse_action(text: str):
    match = re.search(r'Action:\s*(\w+)\("([^"]+)"\)', text)
    if match:
        return match.group(1), match.group(2)
    return None, None

# ── Simulate the ReAct loop ────────────────────────────────────
def simulate_loop(question: str, max_turns: int = 6) -> str:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    messages = [{"role": "user", "content": question}]

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=build_system_prompt(),
            messages=messages
        )
        output = response.content[0].text.strip()
        print(f"\n[Turn {turn}] Claude:\n{output}")

        # Check if agent is done
        if output.startswith("Answer:"):
            print(f"\n✓ Done in {turn} turns.")
            return output

        # Parse and execute the action
        tool_name, tool_input = parse_action(output)
        if not tool_name:
            print("  No valid action found — stopping.")
            break

        observation = fake_tool_call(tool_name, tool_input)
        print(f"[Tool] {tool_name}('{tool_input}') → {observation}")

        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "Max turns reached without an answer."


if __name__ == "__main__":
    simulate_loop("What is the weather in London?")
    simulate_loop("What is the weather in Tokyo and what time is it there?")