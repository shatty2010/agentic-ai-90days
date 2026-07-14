"""
Day 41 — Observability & Tracing for Agents
Week 9: Production, Security & Observability | 90-Day Agentic AI Plan

Span-based tracing wrapped around the Week 8 multi-agent pipeline.
Exports to: console (always), AWS X-Ray (if credentials present),
Azure Application Insights (if connection string present).

Requires:
  pip install opentelemetry-api opentelemetry-sdk azure-monitor-opentelemetry-exporter boto3
  Falls back to console-only export if no cloud credentials are configured.
"""


import os
import time
import uuid
import json
import contextvars
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Span model
# ---------------------------------------------------------------------------

class Span:
    def __init__(self, name: str, parent: "Span | None" = None):
        self.name = name
        self.span_id = uuid.uuid4().hex[:16]
        self.trace_id = parent.trace_id if parent else uuid.uuid4().hex
        self.parent_id = parent.span_id if parent else None
        self.depth = (parent.depth + 1) if parent else 0
        self.start = time.time()
        self.end = None
        self.attributes = {}
        self.children: list["Span"] = []
        if parent is not None:
            parent.children.append(self)

    def set(self, **kwargs) -> "Span":
        self.attributes.update(kwargs)
        return self

    def close(self) -> "Span":
        self.end = time.time()
        return self

    @property
    def duration_ms(self) -> float:
        end = self.end if self.end is not None else time.time()
        return round((end - self.start) * 1000, 1)


# contextvars threads "current span" through nested calls without passing
# a span object through every function signature by hand.
_current_span: contextvars.ContextVar = contextvars.ContextVar("current_span", default=None)


class traced:
    """Context manager: opens a Span as a child of whatever the current span is."""

    def __init__(self, name: str, **attrs):
        self.name = name
        self.attrs = attrs
        self.span = None
        self.token = None

    def __enter__(self) -> Span:
        parent = _current_span.get()
        self.span = Span(self.name, parent=parent).set(**self.attrs)
        self.token = _current_span.set(self.span)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.span.set(error=str(exc_val), error_type=exc_type.__name__)
        self.span.close()
        _current_span.reset(self.token)
        if self.span.parent_id is None:
            # Root span finished — the whole trace is complete, export it.
            SpanExporter().export_all(self.span)
        return False  # never swallow exceptions


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

class SpanExporter:

    def export_all(self, root: Span) -> None:
        self.export_console(root)
        self.export_xray(root)
        self.export_app_insights(root)

    def export_console(self, span: Span, _depth: int = 0) -> None:
        indent = "  " * span.depth
        err = f" [ERROR: {span.attributes.get('error')}]" if "error" in span.attributes else ""
        attrs = {k: v for k, v in span.attributes.items() if k not in ("error", "error_type")}
        attrs_str = f" {attrs}" if attrs else ""
        print(f"{indent}├─ {span.name} ({span.duration_ms}ms){attrs_str}{err}")
        for child in span.children:
            self.export_console(child)

    def export_xray(self, root: Span) -> None:
        try:
            import boto3
            xray = boto3.client("xray", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        except Exception as e:
            print(f"  [X-Ray] skipped: {type(e).__name__}")
            return

        def flatten(span: Span, out: list):
            out.append(span)
            for c in span.children:
                flatten(c, out)

        all_spans = []
        flatten(root, all_spans)
        try:
            docs = []
            for s in all_spans:
                docs.append(json.dumps({
                    "id": s.span_id,
                    "trace_id": f"1-{s.trace_id[:8]}-{s.trace_id[8:24].ljust(24,'0')}",
                    "parent_id": s.parent_id,
                    "name": s.name,
                    "start_time": s.start,
                    "end_time": s.end or s.start,
                    "metadata": {"attrs": {k: str(v) for k, v in s.attributes.items()}},
                }))
            xray.put_trace_segments(TraceSegmentDocuments=docs)
            print(f"  [X-Ray] exported {len(docs)} spans, trace_id={root.trace_id[:12]}...")
        except Exception as e:
            print(f"  [X-Ray] export failed (expected without real AWS creds): {type(e).__name__}: {str(e)[:100]}")

    def export_app_insights(self, root: Span) -> None:
        conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
        if not conn_str:
            print("  [App Insights] APPLICATIONINSIGHTS_CONNECTION_STRING not set — skipping")
            return
        try:
            from opentelemetry import trace as ot_trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

            provider = TracerProvider()
            provider.add_span_processor(BatchSpanProcessor(
                AzureMonitorTraceExporter(connection_string=conn_str)
            ))
            ot_trace.set_tracer_provider(provider)
            tracer = ot_trace.get_tracer(__name__)

            def emit(span: Span):
                with tracer.start_as_current_span(span.name) as otel_span:
                    for k, v in span.attributes.items():
                        otel_span.set_attribute(k, str(v))
                    for c in span.children:
                        emit(c)

            emit(root)
            print(f"  [App Insights] exported trace_id={root.trace_id[:12]}...")
        except Exception as e:
            print(f"  [App Insights] export failed: {type(e).__name__}: {str(e)[:100]}")


# ---------------------------------------------------------------------------
# Traced agent pipeline (mirrors Week 8's supervisor/worker shape)
# ---------------------------------------------------------------------------

_client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"


def call_llm(prompt: str, max_tokens: int = 300) -> str:
    with traced("llm_call", model=MODEL, prompt_len=len(prompt)) as span:
        resp = _client.messages.create(
            model=MODEL, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        span.set(tokens_out=resp.usage.output_tokens, tokens_in=resp.usage.input_tokens)
        return text


def researcher_agent(question: str) -> str:
    with traced("researcher_agent", question=question):
        with traced("tool_call", tool="search_docs"):
            time.sleep(0.05)  # simulated retrieval latency
        return call_llm(f"In two sentences, research: {question}")


def analyser_agent(findings: str) -> str:
    with traced("analyser_agent"):
        return call_llm(f"In one sentence, identify the key insight in: {findings}")


def writer_agent(findings: str, insight: str) -> str:
    with traced("writer_agent"):
        return call_llm(f"Write a one-sentence recommendation given findings '{findings}' and insight '{insight}'")


def run_pipeline(question: str) -> str:
    with traced("run_pipeline", question=question) as root:
        findings = researcher_agent(question)
        insight = analyser_agent(findings)
        answer = writer_agent(findings, insight)
        root.set(final_answer_len=len(answer))
        return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    question = "Should a fintech startup use Bedrock or Azure AI Agent Service for its first production agent?"

    print("\n" + "=" * 64)
    print("TRACED PIPELINE RUN")
    print("=" * 64)
    result = run_pipeline(question)
    print(f"\n  Final answer: {result}")

    print("\n" + "=" * 64)
    print("SPAN TREE (console export happens automatically when the root span closes)")
    print("=" * 64)
