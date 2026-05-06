import time
import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import os
load_dotenv(override=True)
Deepseek_api_key = os.getenv('Deepseek_API_KEY')
start = time.time()
client = anthropic.Client()
message = client.messages.create(
    model="claude-haiku-4-5-20251001",   # cheapest model — use this for all experiments
    max_tokens=256,
    system="You are a talkative assistant.",
    messages=[
        {"role": "user", "content": "What is 2 + 2?"}
    ]
)
latency = round((time.time() - start) * 1000)

print("─── Response ───────────────────────────────")
print(message.content[0])
print()
print("─── Usage ──────────────────────────────────")
print(f"Input tokens  : {message.usage.input_tokens}")
print(f"Output tokens : {message.usage.output_tokens}")
print(f"Latency       : {latency} ms")

start = time.time()
openai_client = OpenAI(api_key=Deepseek_api_key,base_url="https://api.deepseek.com/v1")
model_name = "deepseek-chat"
openai_messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a talkative assistant"},
    {"role": "user", "content": "What is 2 + 2?"}
]
response = openai_client.chat.completions.create(model=model_name, messages=openai_messages)
print("─── Response ───────────────────────────────")
print(response.choices[0].message.content)
print()
print("─── Usage ──────────────────────────────────")
latency = round((time.time() - start) * 1000)
print(f"Latency       : {latency} ms")