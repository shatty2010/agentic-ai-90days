import anthropic
import boto3
import json
import os
from openai import AzureOpenAI

from dotenv import load_dotenv

load_dotenv()

def chat_anthropic(user_message :str,system_message:str ="You are a helpful assistant.") -> dict:
    client = anthropic.Anthropic()
    
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=system_message,
        messages=[{"role": "user", "content": user_message}]
    )
    block = message.content[0]
    if isinstance(block, anthropic.types.TextBlock):
        response_text = block.text.replace("\n", " ").strip()
    else:
        response_text = str(block)

    return {
        "backend": "anthropic",
        "response": response_text,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }

def chat_bedrock(user_message :str, system_message:str ="You are a helpful assistant.") -> dict:
    client = boto3.client('bedrock-runtime', region_name="us-east-1")
    # Prefer an inference-profile ARN if provided (required for some Bedrock models)
    model_identifier = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN") or 'us.anthropic.claude-haiku-4-5-20251001-v1:0'

    try:
        response = client.converse(
            modelId=model_identifier,
            system=[{"text": system_message}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"maxTokens": 256}
        )
    except Exception as e:
        raise RuntimeError(
            f"Bedrock Converse failed: {e}.\nIf this model requires an inference profile, set the environment variable `BEDROCK_INFERENCE_PROFILE_ARN` to the profile ID or ARN that contains the model and retry."
        ) from e
    
    return {
        "backend": "bedrock",
        "response": response["output"]["message"]["content"][0]["text"],
        "input_tokens": response["usage"]["inputTokens"],
        "output_tokens": response["usage"]["outputTokens"],
    }

# ── Azure OpenAI ───────────────────────────────────────────
def chat_azure(user_message: str, system_message: str = "You are a helpful assistant.") -> dict:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-02-01"
    )
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=256
    )
    return {
        "backend": "azure",
        "response": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }

def chat(user_message: str, backend: str = "anthropic", system_message: str = "You are a helpful assistant.") -> dict:
    backends = {
        "anthropic": chat_anthropic,
        "bedrock": chat_bedrock,
        "azure": chat_azure,
    }
    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Choose from: {list(backends.keys())}")
    return backends[backend](user_message, system_message)

# ── Test all 3 ─────────────────────────────────────────────
if __name__ == "__main__":
    prompt = "In one sentence, what is a microservice?"

    for backend in ["anthropic", "azure"]:
        try:
            result = chat(prompt, backend=backend)
            print(f"\n[{result['backend'].upper()}]")
            print(f"  Response : {result['response'].strip()}")
            print(f"  Tokens   : {result['input_tokens']} in / {result['output_tokens']} out")
        except Exception as e:
            print(f"\n[{backend.upper()}] ERROR: {e}")    