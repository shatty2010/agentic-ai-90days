import anthropic
from dotenv import load_dotenv

load_dotenv()

MAX_HISTORY_TOKENS = 1000  # deliberately small to force trimming in tests

class ConversationManager:
    def __init__(self, system_prompt: str, max_tokens: int = MAX_HISTORY_TOKENS):
        self.client = anthropic.Anthropic()
        self.system = system_prompt
        self.max_tokens = max_tokens
        self.history: list[dict] = []

    def _estimate_tokens(self, messages: list[dict]) -> int:
        # rough estimate: 1 token ≈ 4 chars
        total = sum(len(m["content"]) for m in messages)
        return total // 4

    def _trim_history(self):
        while (
            len(self.history) > 2
            and self._estimate_tokens(self.history) > self.max_tokens
        ):
            # drop oldest user+assistant pair (indices 0 and 1)
            self.history.pop(0)
            self.history.pop(0)

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        self._trim_history()

        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=self.system,
            messages=self.history,
        )
        reply = response.content[0].text.strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def print_history(self):
        print(f"\n--- History ({len(self.history)} messages, "
              f"~{self._estimate_tokens(self.history)} tokens) ---")
        for m in self.history:
            label = "You" if m["role"] == "user" else "AI"
            print(f"[{label}] {m['content'][:80]}")


if __name__ == "__main__":
    bot = ConversationManager(
        system_prompt="You are a helpful assistant. Be concise.",
        max_tokens=500,
    )

    exchanges = [
        "My name is Satya and I work at Infosys.",
        "I am learning Agentic AI to get a job abroad.",
        "What is my name?",
        "Where do I work?",
        "What am I learning and why?",
        # These next ones will start pushing older context out if max_tokens is small
        "Tell me a one-sentence fact about Python.",
        "Tell me a one-sentence fact about AWS.",
        "Tell me a one-sentence fact about .NET.",
        "Do you still know my name?",
    ]

    for msg in exchanges:
        print(f"\nYou: {msg}")
        reply = bot.chat(msg)
        print(f"AI:  {reply}")

    bot.print_history()