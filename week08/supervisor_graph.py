"""
Day 39 — LangGraph Multi-Agent Supervisor Graph
Week 8: Multi-Agent Systems | 90-Day Agentic AI Plan

Supervisor node routes to specialist sub-agents (researcher / analyser / writer).
Each agent node uses LLMRouter — swap cloud backend via env var.
"""

import os
import json
import operator
import anthropic
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

claude = anthropic.Anthropic()
MODEL  = "claude-haiku-4-5-20251001"

# Cloud backend config — override via env vars
RESEARCHER_BACKEND = os.getenv("RESEARCHER_BACKEND", "anthropic")
ANALYSER_BACKEND   = os.getenv("ANALYSER_BACKEND",   "anthropic")
WRITER_BACKEND     = os.getenv("WRITER_BACKEND",     "anthropic")


# ---------------------------------------------------------------------------
# Simple LLMRouter (self-contained — no import from Day 38 needed)
# ---------------------------------------------------------------------------

class LLMRouter:
    """Route calls to Anthropic, Bedrock, or Azure. Falls back to Anthropic."""

    def __init__(self, backend: str = "anthropic", region: str = "us-east-1"):
        self.backend = backend
        self._client = anthropic.Anthropic()   # always available as fallback

        if backend == "bedrock":
            try:
                import boto3
                import botocore
                self._bedrock = boto3.client("bedrock-runtime", region_name=region)
                self._bedrock_model = "anthropic.claude-haiku-4-5-20251001:0"
            except Exception:
                self._bedrock = None
        elif backend == "azure":
            try:
                from openai import AzureOpenAI
                self._azure = AzureOpenAI(
                    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                    api_key        = os.getenv("AZURE_OPENAI_API_KEY", ""),
                    api_version    = "2024-05-01-preview",
                )
                self._azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            except Exception:
                self._azure = None

    def chat(self, messages: list, system: str = "", max_tokens: int = 512) -> str:
        try:
            if self.backend == "bedrock" and getattr(self, "_bedrock", None):
                kwargs = {
                    "modelId":         self._bedrock_model,
                    "messages":        [{"role": m["role"],
                                         "content": [{"text": m["content"]}]} for m in messages],
                    "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.3},
                }
                if system:
                    kwargs["system"] = [{"text": system}]
                resp = self._bedrock.converse(**kwargs)
                return resp["output"]["message"]["content"][0]["text"]

            elif self.backend == "azure" and getattr(self, "_azure", None):
                full_messages = []
                if system:
                    full_messages.append({"role": "system", "content": system})
                full_messages.extend(messages)
                resp = self._azure.chat.completions.create(
                    model=self._azure_deployment, messages=full_messages, max_tokens=max_tokens
                )
                return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [LLMRouter:{self.backend}] failed, falling back: {e}")

        # Anthropic fallback
        resp = self._client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=max_tokens,
            system=system, messages=messages,
        )
        return resp.content[0].text.strip()

    def __repr__(self) -> str:
        return f"LLMRouter({self.backend!r})"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    goal:         str
    next_agent:   str
    research:     str
    analysis:     str
    final_report: str
    step_count:   int
    history:      Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are an agent supervisor. Given the current state, decide which agent to run next.

Available agents:
  researcher — gathers facts, data, live information
  analyser   — interprets research, finds patterns and insights
  writer     — produces polished final output from analysis

Rules (follow exactly):
  - If research is empty → return researcher
  - If research is done but analysis is empty → return analyser
  - If analysis is done but final_report is empty → return writer
  - If final_report is populated → return FINISH

Return ONLY ONE word: researcher | analyser | writer | FINISH"""


def supervisor_node(state: SupervisorState) -> dict:
    step = state.get("step_count", 0) + 1

    # Hard safety cap
    if step > 10:
        print("  [supervisor] step cap reached → FINISH")
        return {"next_agent": "FINISH", "step_count": step}

    status = (
        f"Goal: {state['goal']}\n"
        f"Research done: {'yes' if state.get('research') else 'no'}\n"
        f"Analysis done: {'yes' if state.get('analysis') else 'no'}\n"
        f"Report done:   {'yes' if state.get('final_report') else 'no'}"
    )
    resp = claude.messages.create(
        model=MODEL, max_tokens=20,
        system=SUPERVISOR_PROMPT,
        messages=[{"role": "user", "content": status}],
    )
    raw        = resp.content[0].text.strip().lower().split()[0]
    valid      = {"researcher", "analyser", "writer", "finish"}
    next_agent = raw if raw in valid else "FINISH"
    print(f"\n  [supervisor] step {step} → {next_agent}")

    return {
        "next_agent": next_agent,
        "step_count": step,
        "history":    [f"[step {step}] supervisor → {next_agent}"],
    }


def route_from_supervisor(state: SupervisorState) -> Literal["researcher", "analyser", "writer", "__end__"]:
    mapping = {
        "researcher": "researcher",
        "analyser":   "analyser",
        "writer":     "writer",
        "finish":     "__end__",
    }
    return mapping.get(state["next_agent"].lower(), "__end__")


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------

def researcher_node(state: SupervisorState) -> dict:
    router = LLMRouter(backend=RESEARCHER_BACKEND)
    print(f"  [researcher] using {router}")
    research = router.chat(
        messages=[{"role": "user", "content": f"Research goal: {state['goal']}"}],
        system=(
            "You are a Researcher agent. Gather relevant facts, data, and insights "
            "for the given goal. Be specific. 3-4 sentences."
        ),
        max_tokens=512,
    )
    print(f"  [researcher] done ({len(research)} chars)")
    return {
        "research": research,
        "history":  [f"[researcher] {research[:80]}..."],
    }


def analyser_node(state: SupervisorState) -> dict:
    router = LLMRouter(backend=ANALYSER_BACKEND)
    print(f"  [analyser] using {router}")
    analysis = router.chat(
        messages=[{"role": "user", "content":
                   f"Goal: {state['goal']}\n\nResearch:\n{state['research']}\n\n"
                   f"Analyse this research: identify patterns, draw comparisons, surface key insights."}],
        system=(
            "You are an Analyser agent. Interpret the research, identify patterns, "
            "and produce clear insights. 2-3 sentences."
        ),
        max_tokens=384,
    )
    print(f"  [analyser] done ({len(analysis)} chars)")
    return {
        "analysis": analysis,
        "history":  [f"[analyser] {analysis[:280]}..."],
    }


def writer_node(state: SupervisorState) -> dict:
    router = LLMRouter(backend=WRITER_BACKEND)
    print(f"  [writer] using {router}")
    report = router.chat(
        messages=[{"role": "user", "content":
                   f"Goal: {state['goal']}\n\nResearch:\n{state['research']}\n\n"
                   f"Analysis:\n{state['analysis']}\n\n"
                   f"Write the final polished output for this goal."}],
        system=(
            "You are a Writer agent. Produce a clear, professional final output. "
            "Structure it well. No jargon. Directly address the original goal."
        ),
        max_tokens=512,
    )
    print(f"  [writer] done ({len(report)} chars)")
    return {
        "final_report": report,
        "history":      [f"[writer] {report[:280]}..."],
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_supervisor_graph():
    g = StateGraph(SupervisorState)

    g.add_node("supervisor",  supervisor_node)
    g.add_node("researcher",  researcher_node)
    g.add_node("analyser",    analyser_node)
    g.add_node("writer",      writer_node)

    g.add_edge(START, "supervisor")

    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "analyser":   "analyser",
            "writer":     "writer",
            "__end__":    END,
        },
    )

    # All agents route back to supervisor after completing
    g.add_edge("researcher", "supervisor")
    g.add_edge("analyser",   "supervisor")
    g.add_edge("writer",     "supervisor")

    return g.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GOALS = [
    (
        "Research the current weather in Pune and Bangalore, analyse which city is good for outdoor "
        "remote work in July, and write a 2-paragraph recommendation for a software developer."
    )
]

if __name__ == "__main__":
    app = build_supervisor_graph()

    print(f"\nCloud routing: researcher={RESEARCHER_BACKEND}, "
          f"analyser={ANALYSER_BACKEND}, writer={WRITER_BACKEND}")
    print("Override with: export RESEARCHER_BACKEND=bedrock ANALYSER_BACKEND=azure")

    for goal in GOALS:
        print(f"\n{'='*64}")
        print(f"Goal: {goal}")
        print(f"{'='*64}")

        initial: SupervisorState = {
            "goal":         goal,
            "next_agent":   "",
            "research":     "",
            "analysis":     "",
            "final_report": "",
            "step_count":   0,
            "history":      [],
        }
        final = app.invoke(initial)

        print(f"\n  History:")
        for h in final["history"]:
            print(f"    {h}")
        print(f"\n  FINAL REPORT:\n  {final['final_report'][:400]}")
