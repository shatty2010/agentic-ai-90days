import anthropic
import os
from dotenv import load_dotenv

load_dotenv(override=True)

client = anthropic.Client()
temperature_values = [0.0, 0.3, 0.7, 1.0]

print(f"{'Temp':<6} {'Tokens':>6}  {'Response'}")
print("─" * 80)

PROMPT = "Explain what a microservice is. Be concise."


for temp in temperature_values:
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        temperature=temp,
        system="You are a helpful assistant.",
        messages=[
            {"role": "user", "content": PROMPT}
        ]
    )
    block = message.content[0]
    response = block.text.replace("\n", " ").strip() if isinstance(block, anthropic.types.TextBlock) else ""
    tokens = message.usage.output_tokens
    print(f"{temp:<6} {tokens:>6}  {response[:500]}")
