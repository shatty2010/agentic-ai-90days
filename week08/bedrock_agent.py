"""
Day 37 — AWS Bedrock Integration
Week 8: Multi-Agent Systems | 90-Day Agentic AI Plan

Section 1: BedrockLLM — same logic as Anthropic SDK, Bedrock Converse API
Section 2: Bedrock Agents — architecture walkthrough + mock invocation
Section 3: LLMRouter — cloud-agnostic wrapper (Anthropic or Bedrock)

Requires:
  pip install boto3
  AWS credentials configured (aws configure) + Bedrock model access enabled
  Falls back gracefully if credentials are not present.
"""

import os
import json
import anthropic
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Section 1 — Bedrock Runtime (Converse API)
# ---------------------------------------------------------------------------

class BedrockLLM:
    """
    Calls Claude on AWS Bedrock via the Converse API.
    Drop-in replacement for anthropic.Anthropic().messages.create()
    with minor format differences.
    """

    # Cross-region inference profile (us.anthropic.*) or direct model ID
    MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    def __init__(self, region: str = "us-east-1"):
        import boto3
        self.client   = boto3.client("bedrock-runtime", region_name=region)
        self.region   = region
        self.model_id = self.MODEL_ID

    def _to_bedrock_messages(self, messages: list) -> list:
        """
        Convert Anthropic-style messages to Bedrock Converse format.
        Anthropic: {"role": "user", "content": [{"type": "text", "text": "..."}]}
        Bedrock:   {"role": "user", "content": [{"text": "..."}]}
        """
        converted = []
        for msg in messages:
            role    = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                converted.append({"role": role, "content": [{"text": content}]})
                continue

            bedrock_content = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        bedrock_content.append({"text": block["text"]})
                    elif block.get("type") == "tool_result":
                        # Anthropic tool_result → Bedrock toolResult
                        bedrock_content.append({
                            "toolResult": {
                                "toolUseId": block["tool_use_id"],
                                "content":   [{"text": block.get("content", "")}],
                            }
                        })
                    elif block.get("type") == "tool_use":
                        bedrock_content.append({
                            "toolUse": {
                                "toolUseId": block["id"],
                                "name":      block["name"],
                                "input":     block["input"],
                            }
                        })
                elif hasattr(block, "type"):
                    # SDK object (ToolUseBlock, TextBlock, etc.)
                    if block.type == "text":
                        bedrock_content.append({"text": block.text})
                    elif block.type == "tool_use":
                        bedrock_content.append({
                            "toolUse": {
                                "toolUseId": block.id,
                                "name":      block.name,
                                "input":     block.input,
                            }
                        })

            converted.append({"role": role, "content": bedrock_content})
        return converted

    def _to_bedrock_tools(self, tools: list) -> dict:
        """Convert Anthropic tool schema list → Bedrock toolConfig."""
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name":        t["name"],
                        "description": t.get("description", ""),
                        "inputSchema": {"json": t["input_schema"]},
                    }
                }
                for t in tools
            ]
        }

    def chat(self, messages: list, system: str = "", max_tokens: int = 512) -> str:
        """Single-turn or multi-turn chat without tools."""
        kwargs = {
            "modelId":        self.model_id,
            "messages":       self._to_bedrock_messages(messages),
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.3},
        }
        if system:
            kwargs["system"] = [{"text": system}]

        resp = self.client.converse(**kwargs)
        return resp["output"]["message"]["content"][0]["text"]

    def tool_loop(self, system: str, user: str, tools: list, max_steps: int = 6) -> str:
        """Full ReAct tool loop using Bedrock Converse."""
        from week08.multi_agent_basics import TOOL_REGISTRY  # reuse tool functions

        messages    = [{"role": "user", "content": [{"text": user}]}]
        tool_config = self._to_bedrock_tools(tools)

        for _ in range(max_steps):
            kwargs = {
                "modelId":         self.model_id,
                "messages":        messages,
                "inferenceConfig": {"maxTokens": 512, "temperature": 0.3},
                "toolConfig":      tool_config,
            }
            if system:
                kwargs["system"] = [{"text": system}]

            resp   = self.client.converse(**kwargs)
            msg    = resp["output"]["message"]
            stop   = resp["stopReason"]
            messages.append(msg)

            if stop == "end_turn":
                for block in msg["content"]:
                    if "text" in block:
                        return block["text"]
                return ""

            if stop == "tool_use":
                tool_results = []
                for block in msg["content"]:
                    if "toolUse" in block:
                        tu   = block["toolUse"]
                        fn   = TOOL_REGISTRY.get(tu["name"])
                        res  = fn(**tu["input"]) if fn else {"error": "unknown tool"}
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tu["toolUseId"],
                                "content":   [{"text": json.dumps(res)}],
                            }
                        })
                messages.append({"role": "user", "content": tool_results})

        return "Max steps reached."


# ---------------------------------------------------------------------------
# Section 2 — Bedrock Agents (architecture + mock invocation)
# ---------------------------------------------------------------------------

def show_bedrock_agent_config() -> None:
    """
    Print the configuration you would set up in the AWS Console or CDK
    to create a Bedrock Agent equivalent to your Day 36 supervisor.
    """
    config = {
        "Agent": {
            "agentName":       "multi-agent-supervisor",
            "foundationModel": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "instruction": (
                "You are a supervisor agent. Decompose the user's goal into tasks "
                "and delegate to the appropriate action group. Synthesise a final answer."
            ),
            "idleSessionTTLInSeconds": 3600,
        },
        "ActionGroups": [
            {
                "actionGroupName": "ResearchTools",
                "description":     "Tools for gathering live data",
                "actionGroupExecutor": {
                    "lambda": "arn:aws:lambda:us-east-1:123456789012:function:research-tools"
                },
                "functionSchema": {
                    "functions": [
                        {
                            "name":        "get_weather",
                            "description": "Get current weather for a city",
                            "parameters": {
                                "city": {"type": "string", "description": "City name", "required": True}
                            },
                        },
                        {
                            "name":        "calculate",
                            "description": "Evaluate a math expression",
                            "parameters": {
                                "expression": {"type": "string", "description": "Math expression", "required": True}
                            },
                        },
                    ]
                },
            }
        ],
        "KnowledgeBase": {
            "knowledgeBaseId":          "XXXXXXXXXX",
            "description":              "Agentic AI document corpus",
            "knowledgeBaseState":       "ENABLED",
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {"numberOfResults": 5}
            },
        },
    }
    print("\n  [Bedrock Agent Config — what you'd set up in console/CDK]")
    print(json.dumps(config, indent=2))


def mock_invoke_agent(agent_id: str = "XXXXXXXXXX",
                       session_id: str = "session-001",
                       user_input: str = "What is the weather in Singapore?") -> None:
    """
    Shows the exact boto3 code to invoke a Bedrock Agent.
    Requires a real agentId and agentAliasId to run.
    """
    print(f"\n  [mock_invoke_agent] agent_id={agent_id}, session_id={session_id}")
    print(f"  Input: {user_input}")
    print("""
  # Real invocation code:
  import boto3
  bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

  response = bedrock_agent.invoke_agent(
      agentId="XXXXXXXXXX",
      agentAliasId="XXXXXXXXXX",    # "TSTALIASID" for draft alias during testing
      sessionId=session_id,
      inputText=user_input,
  )

  # EventStream iteration — response streams in chunks
  full_response = ""
  for event in response["completion"]:
      if "chunk" in event:
          chunk = event["chunk"]["bytes"].decode("utf-8")
          full_response += chunk
      if "trace" in event:
          # Orchestration trace: shows tool calls, reasoning steps
          print("  [trace]", event["trace"])
      if "returnControl" in event:
          # Agent paused for external input (HITL equivalent)
          print("  [returnControl]", event["returnControl"])

  print("Agent response:", full_response)
  """)

    # Actually try — will fail gracefully without real credentials/agentId
    try:
        import boto3
        client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        resp   = client.invoke_agent(
            agentId=agent_id, agentAliasId="TSTALIASID",
            sessionId=session_id, inputText=user_input,
        )
        full = ""
        for event in resp["completion"]:
            if "chunk" in event:
                full += event["chunk"]["bytes"].decode()
        print(f"  Agent response: {full[:200]}")
    except Exception as e:
        print(f"  [Expected] Cannot invoke without real agent: {type(e).__name__}: {str(e)[:1000]}")


# ---------------------------------------------------------------------------
# Section 3 — Cloud-agnostic LLM Router
# ---------------------------------------------------------------------------

class LLMRouter:
    """
    Route LLM calls to either Anthropic SDK or AWS Bedrock Runtime.
    Swap backend with one config change — call code stays identical.

    .NET analogy: IHttpClientFactory with named clients.
    """

    def __init__(self, backend: Literal["anthropic", "bedrock"] = "anthropic",
                 region: str = "us-east-1"):
        self.backend = backend
        if backend == "bedrock":
            self._bedrock = BedrockLLM(region=region)
        else:
            self._anthropic = anthropic.Anthropic()

    def chat(self, messages: list[dict], system: str = "", max_tokens: int = 512) -> str:
        if self.backend == "bedrock":
            return self._bedrock.chat(messages, system=system, max_tokens=max_tokens)
        else:
            resp = self._anthropic.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return resp.content[0].text.strip()

    def __repr__(self) -> str:
        return f"LLMRouter(backend={self.backend!r})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    "What is the current weather like in Sydney?",
]

if __name__ == "__main__":

    # --- Section 1: Bedrock Runtime ---
    print("\n" + "="*64)
    print("SECTION 1 — Bedrock Runtime (Converse API)")
    print("="*64)
    try:
        bedrock_llm = BedrockLLM(region="us-east-1")
        for q in TEST_QUESTIONS:
            print(f"\n  Q: {q}")
            answer = bedrock_llm.chat([{"role": "user", "content": q}],
                                       system="Be concise. Two sentences max.")
            print(f"  A (Bedrock): {answer}")
    except Exception as e:
        print(f"  [Skipped] Bedrock credentials not configured: {type(e).__name__}")
        print("  Configure with: aws configure (region us-east-1, enable Claude Haiku in Bedrock)")

    # --- Section 2: Bedrock Agents ---
    print("\n" + "="*64)
    print("SECTION 2 — Bedrock Agents (architecture + mock)")
    print("="*64)
    show_bedrock_agent_config()
    mock_invoke_agent()

    # --- Section 3: LLM Router ---
    print("\n" + "="*64)
    print("SECTION 3 — LLMRouter (cloud-agnostic)")
    print("="*64)

    q = TEST_QUESTIONS[0]
    msgs = [{"role": "user", "content": q}]

    for backend in ("anthropic", "bedrock"):
        router = LLMRouter(backend=backend)
        print(f"\n  Router: {router}")
        print(f"  Q: {q}")
        try:
            answer = router.chat(msgs, system="Be concise. Two sentences max.")
            print(f"  A: {answer}")
        except Exception as e:
            print(f"  [Skipped] {type(e).__name__}: {str(e)[:1000]}")
