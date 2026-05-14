import anthropic, json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# --- Tool definitions ---
TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluates a basic math expression. Use for any arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression, e.g. '2 ** 10' or '(45 * 12) / 7'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Returns current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "search_web",
        "description": "Returns a brief search result snippet for a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                }
            },
            "required": ["query"]
        }
    }
]

# --- Tool implementations (stubs) ---
def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    data = {
        "mumbai": "32°C, humid, partly cloudy",
        "delhi": "38°C, hot, clear sky",
        "bangalore": "27°C, pleasant, light breeze",
    }
    return data.get(city.lower(), f"No data for {city}")

def search_web(query: str) -> str:
    return f"[Stub result] Top result for '{query}': Wikipedia article found. No live data in this demo."

TOOL_MAP = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_web": search_web,
}

# --- ReAct agent with memory ---
class ReactAgent:
    def __init__(self):
        self.history: list[dict] = []
        self.system = (
            "You are a helpful assistant with access to tools. "
            "Think step by step. Use tools when needed. "
            "Give a concise final answer once you have enough information."
        )

    def _trim(self):
        # Keep last 10 messages max (simple cap for this build)
        if len(self.history) > 2:
            self.history = self.history[-2:]

    def run(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        self._trim()

        while True:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                system=self.system,
                tools=TOOLS,
                messages=self.history,
            )

            # Append assistant response to history
            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Final text answer
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text.strip()

            elif response.stop_reason == "tool_use":
                # Execute each tool call, collect results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        fn = TOOL_MAP.get(block.name)
                        if fn:
                            result = fn(**block.input)
                        else:
                            result = f"Unknown tool: {block.name}"
                        print(f"  [Tool: {block.name}] Input: {block.input} → {result}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                # Feed results back into history
                self.history.append({"role": "user", "content": tool_results})

            else:
                return "[Agent stopped unexpectedly]"


if __name__ == "__main__":
    agent = ReactAgent()
    questions = [
        "What is 2 to the power of 16?",
        "What's the weather in London?",
        "Search for Claude Anthropic API",
        "If I have 3 servers each costing $47.50/month, what's my annual cost?",
        "What city did I ask about the weather earlier?",  # memory test
    ]

    for q in questions:
        print(f"\nYou: {q}")
        answer = agent.run(q)
        print(f"Agent: {answer}")