"""
Day 40 — Capstone: 3-Agent Pipeline with Cloud Routing
Week 8: Multi-Agent Systems | 90-Day Agentic AI Plan

Researcher  → AWS Bedrock   (RESEARCHER_BACKEND=bedrock)
Analyser    → Azure OpenAI  (ANALYSER_BACKEND=azure)
Writer      → Anthropic     (WRITER_BACKEND=anthropic)

All fall back to Anthropic SDK if cloud credentials are absent.
"""

import os
import json
import operator
import anthropic
import requests
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

claude = anthropic.Anthropic()
MODEL  = "claude-haiku-4-5-20251001"

# Cloud backend config
RESEARCHER_BACKEND = os.getenv("RESEARCHER_BACKEND", "bedrock")
ANALYSER_BACKEND   = os.getenv("ANALYSER_BACKEND",   "azure")
WRITER_BACKEND     = os.getenv("WRITER_BACKEND",     "anthropic")


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

class PipelineReport(BaseModel):
    goal:              str
    research_summary:  str
    key_insights:      list[str] = Field(min_length=2, max_length=4)
    final_answer:      str
    cloud_routing:     dict[str, str]   # agent → backend used
    total_steps:       int


SUBMIT_REPORT_TOOL = {
    "name": "submit_report",
    "description": "Submit the final structured pipeline report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "research_summary": {"type": "string"},
            "key_insights":     {"type": "array", "items": {"type": "string"},
                                 "minItems": 2, "maxItems": 4},
            "final_answer":     {"type": "string"},
        },
        "required": ["research_summary", "key_insights", "final_answer"],
    },
}


# ---------------------------------------------------------------------------
# Cloud-agnostic LLM Router
# ---------------------------------------------------------------------------

class LLMRouter:
    """Route to Bedrock, Azure OpenAI, or Anthropic SDK. Falls back gracefully."""

    def __init__(self, backend: str = "anthropic", region: str = "us-east-1"):
        self.backend      = backend
        self.active_backend = backend
        self._anthropic   = anthropic.Anthropic()

        self._bedrock = None
        self._azure   = None

        if backend == "bedrock":
            try:
                import boto3
                client = boto3.client("bedrock-runtime", region_name=region)
                # Quick check — list is not needed, just validate client init
                self._bedrock       = client
                self._bedrock_model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
            except Exception as e:
                print(f"    [LLMRouter] Bedrock init failed ({e}), will fallback")

        elif backend == "azure":
            try:
                from openai import AzureOpenAI
                ep = os.getenv("AZURE_OPENAI_ENDPOINT", "")
                ak = os.getenv("AZURE_OPENAI_API_KEY", "")
                if ep and ak:
                    self._azure            = AzureOpenAI(
                        azure_endpoint=ep, api_key=ak, api_version="2024-05-01-preview"
                    )
                    self._azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
                else:
                    print("    [LLMRouter] Azure env vars not set, will fallback")
            except Exception as e:
                print(f"    [LLMRouter] Azure init failed ({e}), will fallback")

    def chat(self, messages: list, system: str = "", max_tokens: int = 512) -> tuple[str, str]:
        """Returns (response_text, backend_actually_used)."""
        if self.backend == "bedrock" and self._bedrock:
            try:
                kwargs = {
                    "modelId":         self._bedrock_model,
                    "messages":        [{"role": m["role"],
                                         "content": [{"text": m["content"]}]}
                                        for m in messages],
                    "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.3},
                }
                if system:
                    kwargs["system"] = [{"text": system}]
                resp = self._bedrock.converse(**kwargs)
                return resp["output"]["message"]["content"][0]["text"], "bedrock"
            except Exception as e:
                print(f"    [bedrock] failed: {e} — using anthropic fallback")

        if self.backend == "azure" and self._azure:
            try:
                full = []
                if system:
                    full.append({"role": "system", "content": system})
                full.extend(messages)
                resp = self._azure.chat.completions.create(
                    model=self._azure_deployment, messages=full, max_tokens=max_tokens
                )
                return resp.choices[0].message.content.strip(), "azure"
            except Exception as e:
                print(f"    [azure] failed: {e} — using anthropic fallback")

        # Anthropic fallback
        resp = self._anthropic.messages.create(
            model=MODEL, max_tokens=max_tokens, system=system, messages=messages
        )
        return resp.content[0].text.strip(), "anthropic"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    goal:         str
    next_agent:   str
    research:     str
    analysis:     str
    final_report: str
    step_count:   int
    cloud_log:    Annotated[list[str], operator.add]   # "agent:backend" per step


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are a pipeline supervisor. Decide which agent to run next.

Rules (in order — return ONLY the one word):
  - If research is empty        → researcher
  - If analysis is empty        → analyser
  - If final_report is empty    → writer
  - If final_report is set      → FINISH

Return exactly one of: researcher | analyser | writer | FINISH"""


def supervisor_node(state: PipelineState) -> dict:
    step = state.get("step_count", 0) + 1
    if step > 12:
        return {"next_agent": "FINISH", "step_count": step}

    status = (
        f"Goal: {state['goal']}\n"
        f"research done: {'yes' if state.get('research') else 'no'}\n"
        f"analysis done: {'yes' if state.get('analysis') else 'no'}\n"
        f"report done:   {'yes' if state.get('final_report') else 'no'}"
    )
    resp = claude.messages.create(
        model=MODEL, max_tokens=20,
        system=SUPERVISOR_PROMPT,
        messages=[{"role": "user", "content": status}],
    )
    raw   = resp.content[0].text.strip().lower().split()[0]
    valid = {"researcher", "analyser", "writer", "finish"}
    next_ = raw if raw in valid else "FINISH"
    print(f"\n  [supervisor] step {step} → {next_}")
    return {"next_agent": next_, "step_count": step}


def route_supervisor(state: PipelineState) -> Literal["researcher", "analyser", "writer", "__end__"]:
    m = {"researcher": "researcher", "analyser": "analyser",
         "writer": "writer", "finish": "__end__"}
    return m.get(state["next_agent"].lower(), "__end__")


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------

def researcher_node(state: PipelineState) -> dict:
    router = LLMRouter(backend=RESEARCHER_BACKEND)
    text, backend = router.chat(
        messages=[{"role": "user", "content": f"Research task: {state['goal']}"}],
        system=(
            "You are a Researcher agent. Gather relevant facts and data for the task. "
            "Be specific and factual. 3-4 sentences."
        ),
        max_tokens=512,
    )
    print(f"  [researcher:{backend}] {len(text)} chars")
    return {"research": text, "cloud_log": [f"researcher:{backend}"]}


def analyser_node(state: PipelineState) -> dict:
    router = LLMRouter(backend=ANALYSER_BACKEND)
    text, backend = router.chat(
        messages=[{"role": "user", "content":
                   f"Goal: {state['goal']}\n\nResearch:\n{state['research']}\n\n"
                   f"Analyse: identify patterns, insights, and key conclusions."}],
        system=(
            "You are an Analyser agent. Identify patterns, draw comparisons, "
            "surface key insights. 2-3 sentences, concrete and specific."
        ),
        max_tokens=384,
    )
    print(f"  [analyser:{backend}] {len(text)} chars")
    return {"analysis": text, "cloud_log": [f"analyser:{backend}"]}


def writer_node(state: PipelineState) -> dict:
    router = LLMRouter(backend=WRITER_BACKEND)
    text, backend = router.chat(
        messages=[{"role": "user", "content":
                   f"Goal: {state['goal']}\n\nResearch:\n{state['research']}\n\n"
                   f"Analysis:\n{state['analysis']}\n\nWrite the final answer."}],
        system=(
            "You are a Writer agent. Produce a clear, professional final answer. "
            "Directly address the goal. Well-structured, no jargon."
        ),
        max_tokens=512,
    )
    print(f"  [writer:{backend}] {len(text)} chars")
    return {"final_report": text, "cloud_log": [f"writer:{backend}"]}


# ---------------------------------------------------------------------------
# Reporter (structured output — runs after graph completes)
# ---------------------------------------------------------------------------

def build_structured_report(state: PipelineState) -> PipelineReport:
    """Ask Claude to fill PipelineReport via submit_report tool."""
    content = (
        f"Goal: {state['goal']}\n\n"
        f"Research:\n{state['research']}\n\n"
        f"Analysis:\n{state['analysis']}\n\n"
        f"Final answer:\n{state['final_report']}\n\n"
        "Call submit_report with the structured output."
    )
    resp = claude.messages.create(
        model=MODEL, max_tokens=512,
        system="Fill submit_report with the pipeline results.",
        tools=[SUBMIT_REPORT_TOOL],
        messages=[{"role": "user", "content": content}],
    )

    # Parse cloud routing from cloud_log
    cloud_routing: dict[str, str] = {}
    for entry in state.get("cloud_log", []):
        if ":" in entry:
            agent, backend = entry.split(":", 1)
            cloud_routing[agent] = backend

    for block in resp.content:
        if block.type == "tool_use" and block.name == "submit_report":
            try:
                return PipelineReport(
                    goal          = state["goal"],
                    cloud_routing = cloud_routing,
                    total_steps   = state.get("step_count", 0),
                    **block.input,
                )
            except ValidationError as e:
                print(f"  [reporter] validation error: {e}")

    # Fallback
    return PipelineReport(
        goal              = state["goal"],
        research_summary  = state["research"][:200],
        key_insights      = [state["analysis"][:100], state["final_report"][:100]],
        final_answer      = state["final_report"],
        cloud_routing     = cloud_routing,
        total_steps       = state.get("step_count", 0),
    )


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("supervisor",  supervisor_node)
    g.add_node("researcher",  researcher_node)
    g.add_node("analyser",    analyser_node)
    g.add_node("writer",      writer_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor", route_supervisor,
        {"researcher": "researcher", "analyser": "analyser",
         "writer": "writer", "__end__": END},
    )
    g.add_edge("researcher", "supervisor")
    g.add_edge("analyser",   "supervisor")
    g.add_edge("writer",     "supervisor")

    return g.compile()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(goal: str) -> PipelineReport:
    print(f"\n{'='*64}")
    print(f"Goal: {goal}")
    print(f"{'='*64}")
    print(f"Cloud routing: researcher={RESEARCHER_BACKEND}, "
          f"analyser={ANALYSER_BACKEND}, writer={WRITER_BACKEND}")

    app = build_graph()
    initial: PipelineState = {
        "goal": goal, "next_agent": "", "research": "", "analysis": "",
        "final_report": "", "step_count": 0, "cloud_log": [],
    }
    final = app.invoke(initial)

    print("\n[Structured output phase]")
    report = build_structured_report(final)
    return report


GOALS = [
    (
        "Research the current weather in Singapore and Sydney, analyse which is better "
        "for a team offsite in August, and write a 1-paragraph recommendation."
    ),
]

if __name__ == "__main__":
    for goal in GOALS:
        report = run_pipeline(goal)
        print(f"\n  PipelineReport:")
        print(f"  cloud_routing:    {report.cloud_routing}")
        print(f"  total_steps:      {report.total_steps}")
        print(f"  research_summary: {report.research_summary[:120]}...")
        print(f"  key_insights:     {report.key_insights}")
        print(f"  final_answer:     {report.final_answer[:300]}...")
        print()
