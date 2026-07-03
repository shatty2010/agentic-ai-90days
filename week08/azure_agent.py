"""
Day 38 — Azure AI Agent Service
Week 8: Multi-Agent Systems | 90-Day Agentic AI Plan

Section 1: AzureLLM — Azure OpenAI via openai SDK
Section 2: Azure AI Agent Service — threads, runs, function tools
Section 3: LLMRouter extended to support "anthropic" | "bedrock" | "azure"

Requires:
  pip install openai azure-ai-projects azure-identity
  Set env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
  Optional:     AZURE_AI_PROJECT_CONN_STR (for full agent service)
  Falls back gracefully if Azure credentials are absent.
"""

import os
import json
import math as _math
from numpy import rint
import requests
import anthropic
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Shared tool functions (reused from earlier weeks)
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=5,
        ).json()
        if not geo.get("results"):
            return json.dumps({"error": f"City not found: {city}"})
        loc = geo["results"][0]
        cw  = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                    "current_weather": True}, timeout=5,
        ).json()["current_weather"]
        return json.dumps({"city": city, "temp_celsius": cw["temperature"],
                           "wind_kph": cw["windspeed"]})
    except Exception as e:
        return json.dumps({"error": str(e)})


def calculate(expression: str) -> str:
    if not all(c in set("0123456789+-*/(). eE") for c in expression):
        return json.dumps({"error": "unsafe expression"})
    try:
        result = eval(expression, {"__builtins__": {}, "math": _math})  # noqa
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


PYTHON_TOOLS = {"get_weather": get_weather, "calculate": calculate}

# OpenAI-format tool schemas
OPENAI_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name":        "get_weather",
            "description": "Get current weather for a city.",
            "parameters":  {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required":   ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "calculate",
            "description": "Evaluate a safe mathematical expression.",
            "parameters":  {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required":   ["expression"],
            },
        },
    },
]

# Responses API format — flat (no "function" nesting), used by client.responses.create()
RESPONSES_TOOL_SCHEMAS = [
    {
        "type":        "function",
        "name":        t["function"]["name"],
        "description": t["function"].get("description", ""),
        "parameters":  t["function"]["parameters"],
    }
    for t in OPENAI_TOOL_SCHEMAS
]


# ---------------------------------------------------------------------------
# Section 1 — Azure OpenAI (direct model call)
# ---------------------------------------------------------------------------

class AzureLLM:
    """
    Calls an Azure-hosted model via the openai SDK.
    Falls back to Anthropic SDK if Azure credentials are not configured.
    """

    def __init__(self):
        self.endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.api_key    = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self.available  = bool(self.endpoint and self.api_key)

        if self.available:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint = self.endpoint,
                api_key        = self.api_key,
                api_version    = "2025-01-01-preview",
            )
            print(f"  [AzureLLM] Connected to: {self.endpoint[:40]}...")
        else:
            self._fallback = anthropic.Anthropic()
            print("  [AzureLLM] Azure not configured — using Anthropic SDK as fallback")

    def chat(self, messages: list, system: str = "", max_tokens: int = 512) -> str:
        if self.available:
            full_messages = []
            if system:
                full_messages.append({"role": "system", "content": system})
            full_messages.extend(messages)
            resp = self._client.chat.completions.create(
                model      = self.deployment,
                messages   = full_messages,
                max_tokens = max_tokens,
            )
            return resp.choices[0].message.content.strip()
        else:
            resp = self._fallback.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=max_tokens,
                system=system, messages=messages,
            )
            return resp.content[0].text.strip()

    def tool_loop(self, system: str, user: str, max_steps: int = 6) -> str:
        """OpenAI-style function calling loop."""
        if not self.available:
            # Fallback to Anthropic tool loop
            from week08.multi_agent_basics import _tool_loop, TOOL_SCHEMAS
            return _tool_loop(system=system, user=user)

        from openai import AzureOpenAI
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]

        for _ in range(max_steps):
            resp    = self._client.chat.completions.create(
                model      = self.deployment,
                messages   = messages,
                tools      = OPENAI_TOOL_SCHEMAS,
                tool_choice= "auto",
                max_tokens = 512,
            )
            choice  = resp.choices[0]
            messages.append(choice.message)

            if choice.finish_reason == "stop":
                return choice.message.content or ""

            if choice.finish_reason == "tool_calls":
                for tc in choice.message.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    fn      = PYTHON_TOOLS.get(fn_name)
                    result  = fn(**fn_args) if fn else json.dumps({"error": "unknown"})
                    print(f"    [tool] {fn_name}({fn_args}) → {result[:80]}")
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })

        return "Max steps reached."


# ---------------------------------------------------------------------------
# Section 2 — Azure AI Agent Service
# ---------------------------------------------------------------------------

def run_azure_agent(question: str) -> str:
    """
    Azure AI Agent Service via Responses API (replaces deprecated Assistants API).
    Stateless: input → tool calls → outputs → final response, chained via previous_response_id.
    """
    endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key    = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    if not endpoint or not api_key:
        print("  [Azure AI Agent] credentials not set — showing code only")
        _show_azure_agent_code(question)
        return "Azure AI Agent: credentials not configured"

    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint = endpoint,
            api_key        = api_key,
            api_version    = "2025-03-01-preview",
        )

        response = client.responses.create(
            model        = deployment,
            instructions = (
                "You are a research assistant. Use tools to gather live data when needed. "
                "Be specific and concise in your answers."
            ),
            input = question,
            tools = RESPONSES_TOOL_SCHEMAS,  # type: ignore[arg-type]
        )
        print(f"  [Azure Agent] id: {response.id}  status: {response.status}")

        # Tool call loop — chain responses via previous_response_id
        for _ in range(6):
            tool_calls = [item for item in response.output if item.type == "function_call"]
            if not tool_calls:
                break
            tool_outputs = []
            for tc in tool_calls:
                fn   = PYTHON_TOOLS.get(tc.name)
                args = json.loads(tc.arguments)
                res  = fn(**args) if fn else json.dumps({"error": "unknown"})
                print(f"    [tool] {tc.name}({args}) → {res[:80]}")
                tool_outputs.append({
                    "type":    "function_call_output",
                    "call_id": tc.call_id,
                    "output":  res,
                })
            response = client.responses.create(
                model                = deployment,
                previous_response_id = response.id,
                input                = tool_outputs,
                tools                = RESPONSES_TOOL_SCHEMAS,  # type: ignore[arg-type]
            )
            print(f"  [Azure Agent] status: {response.status}")

        # Extract text from message output
        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        return block.text
        return ""

    except Exception as e:
        print(f"  [Azure Agent Error] {type(e).__name__}: {str(e)[:120]}")
        return f"Error: {e}"


def run_azure_agent_manual_loop(question: str) -> str:
    """
    Manual tool-call loop via Responses API — mirrors what create_and_process_run did,
    but using the non-deprecated Responses API with explicit per-turn control.
    """
    endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key    = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    if not endpoint or not api_key:
        print("  [Manual loop] credentials not set — skipping")
        return "Manual loop: credentials not configured"

    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint = endpoint,
            api_key        = api_key,
            api_version    = "2025-03-01-preview",
        )

        response = client.responses.create(
            model        = deployment,
            instructions = "Answer questions. Use tools when you need live data.",
            input        = question,
            tools        = RESPONSES_TOOL_SCHEMAS,  # type: ignore[arg-type]
        )
        print(f"    [Manual loop] id: {response.id}  status: {response.status}")

        for step in range(6):
            tool_calls = [item for item in response.output if item.type == "function_call"]
            if not tool_calls:
                break
            tool_outputs = []
            for tc in tool_calls:
                fn   = PYTHON_TOOLS.get(tc.name)
                args = json.loads(tc.arguments)
                res  = fn(**args) if fn else '{"error": "unknown"}'
                print(f"    [step {step+1}] {tc.name}({args}) → {res[:60]}")
                tool_outputs.append({
                    "type":    "function_call_output",
                    "call_id": tc.call_id,
                    "output":  res,
                })
            response = client.responses.create(
                model                = deployment,
                previous_response_id = response.id,
                input                = tool_outputs,
                tools                = RESPONSES_TOOL_SCHEMAS,  # type: ignore[arg-type]
            )
            print(f"    [Manual loop] status: {response.status}")

        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        return block.text

    except Exception as e:
        print(f"  [Manual loop error] {type(e).__name__}: {str(e)[:120]}")

    return ""


def _show_azure_agent_code(question: str) -> None:
    print(f"""
  # Azure AI Agent Service — full flow (run after configuring AZURE_AI_PROJECT_CONN_STR):

  from azure.ai.projects import AIProjectClient
  from azure.ai.projects.models import FunctionTool, ToolSet
  from azure.identity import DefaultAzureCredential

  project = AIProjectClient.from_connection_string(
      conn_str   = os.getenv("AZURE_AI_PROJECT_CONN_STR"),
      credential = DefaultAzureCredential(),
  )
  functions = FunctionTool(functions={{get_weather, calculate}})
  toolset   = ToolSet(); toolset.add(functions)
  agent     = project.agents.create_agent(model="gpt-4o", name="agent",
              instructions="...", toolset=toolset)
  thread    = project.agents.create_thread()
  project.agents.create_message(thread_id=thread.id, role="user", content={question!r})
  run       = project.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
  messages  = project.agents.list_messages(thread_id=thread.id)
  # → extract assistant message from messages.data
    """)


# ---------------------------------------------------------------------------
# Section 3 — Extended LLMRouter (anthropic | bedrock | azure)
# ---------------------------------------------------------------------------

class LLMRouter:
    """
    Route calls to Anthropic SDK, AWS Bedrock Runtime, or Azure OpenAI.
    Identical .chat() interface regardless of backend.

    .NET analogy: IHttpClientFactory with named clients — swap backend,
    not call code.
    """

    def __init__(self, backend: Literal["anthropic", "bedrock", "azure"] = "anthropic",
                 region: str = "us-east-1"):
        self.backend = backend
        if backend == "azure":
            self._azure = AzureLLM()
        elif backend == "bedrock":
            try:
                from week08.bedrock_agent import BedrockLLM
                self._bedrock = BedrockLLM(region=region)
            except Exception as e:
                print(f"  [LLMRouter] Bedrock unavailable: {e}")
                self._bedrock = None
        else:
            self._anthropic = anthropic.Anthropic()

    def chat(self, messages: list, system: str = "", max_tokens: int = 512) -> str:
        if self.backend == "azure":
            return self._azure.chat(messages, system=system, max_tokens=max_tokens)
        elif self.backend == "bedrock":
            if self._bedrock:
                return self._bedrock.chat(messages, system=system, max_tokens=max_tokens)
            # Fallback
            resp = anthropic.Anthropic().messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=max_tokens,
                system=system, messages=messages,
            )
            return resp.content[0].text.strip()
        else:
            resp = self._anthropic.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=max_tokens,
                system=system, messages=messages,
            )
            return resp.content[0].text.strip()

    def tool_loop(self, system: str, user: str, max_steps: int = 6) -> str:
        """Route a tool-capable request to the appropriate backend."""
        if self.backend == "azure":
            return self._azure.tool_loop(system=system, user=user, max_steps=max_steps)
        elif self.backend == "bedrock" and self._bedrock:
            return self._bedrock.tool_loop(system=system, user=user,
                                           tools=OPENAI_TOOL_SCHEMAS, max_steps=max_steps)
        else:
            from week08.multi_agent_basics import _tool_loop
            return _tool_loop(system=system, user=user)

    def __repr__(self) -> str:
        return f"LLMRouter(backend={self.backend!r})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    "What is the weather in Singapore right now?",
]

if __name__ == "__main__":

    # --- Section 1: Azure OpenAI ---
    print("\n" + "="*64)
    print("SECTION 1 — Azure OpenAI (direct model call)")
    print("="*64)
    azure_llm = AzureLLM()
    for q in TEST_QUESTIONS:
        print(f"\n  Q: {q}")
        ans = azure_llm.tool_loop(
            system="You are a research assistant. Use tools when needed. Be concise.",
            user=q,
        )
        print(f"  A: {ans[:200]}")

    # --- Section 2: Azure AI Agent ---
    print("\n" + "="*64)
    print("SECTION 2 — Azure AI Agent Service")
    print("="*64)
    for q in TEST_QUESTIONS:
        print(f"\n  Q: {q}")
        ans = run_azure_agent(q)
        print(f"  A: {ans[:200]}")

    # --- Section 3: Extended LLMRouter ---
    print("\n" + "="*64)
    print("SECTION 3 — LLMRouter (anthropic | azure)")
    print("="*64)
    q    = TEST_QUESTIONS[0]
    msgs = [{"role": "user", "content": q}]

    for backend in ("anthropic", "azure"):
        router = LLMRouter(backend=backend)
        print(f"\n  Router: {router}")
        try:
            ans = router.tool_loop(
                system="You are a research assistant. Use tools for live data. Be concise.",
                user=q,
            )
            print(f"  A: {ans[:150]}")
        except Exception as e:
            print(f"  [Error] {type(e).__name__}: {str(e)[:80]}")
