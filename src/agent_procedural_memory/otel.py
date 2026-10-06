from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from .models import AgentTrace, Span


def _first(mapping: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _normalize_status(raw: Any) -> str:
    if isinstance(raw, Mapping):
        raw = _first(raw, ("code", "status_code", "message"), "OK")
    text = str(raw or "OK").upper()
    if text in {"0", "OK", "STATUS_CODE_OK", "UNSET", "STATUS_CODE_UNSET"}:
        return "OK"
    if text in {"1", "ERROR", "STATUS_CODE_ERROR"}:
        return "ERROR"
    return text


def _normalize_span(raw: dict[str, Any]) -> Span:
    attrs = dict(raw.get("attributes") or {})
    status = _normalize_status(_first(raw, ("status_code", "status", "otel.status_code"), "OK"))
    events = tuple(dict(event) for event in raw.get("events") or ())
    return Span(
        name=str(raw.get("name") or attrs.get("name") or "span"),
        span_id=raw.get("span_id") or raw.get("spanId"),
        parent_span_id=raw.get("parent_span_id") or raw.get("parentSpanId"),
        start_time=raw.get("start_time") or raw.get("startTimeUnixNano"),
        end_time=raw.get("end_time") or raw.get("endTimeUnixNano"),
        status_code=status,
        attributes=attrs,
        events=events,
    )


def normalize_otel_trace(raw: dict[str, Any]) -> AgentTrace:
    """Normalize a common OpenTelemetry JSON trace record into an AgentTrace."""

    attrs = dict(raw.get("attributes") or {})
    spans = tuple(_normalize_span(span) for span in raw.get("spans") or [])
    trace_id = str(raw.get("trace_id") or raw.get("traceId") or attrs.get("trace_id") or "trace")

    span_attrs = [span.attributes for span in spans]
    task_name = str(
        raw.get("task_name")
        or attrs.get("task.name")
        or attrs.get("benchmark.task")
        or _first_in_attrs(span_attrs, ("task.name", "benchmark.task", "workflow.name"), "agent_workflow")
    )
    outcome = str(
        raw.get("outcome")
        or attrs.get("task.outcome")
        or attrs.get("agent.outcome")
        or _first_in_attrs(span_attrs, ("task.outcome", "agent.outcome"), None)
        or _infer_outcome(attrs, spans)
    ).lower()
    return AgentTrace(
        trace_id=trace_id,
        task_name=task_name,
        outcome="success" if outcome in {"success", "succeeded", "pass", "passed", "true"} else "failure",
        spans=spans,
        attributes=attrs,
    )


def _first_in_attrs(attrs: Iterable[dict[str, Any]], keys: Iterable[str], default: Any = None) -> Any:
    for attr in attrs:
        value = _first(attr, keys, None)
        if value is not None:
            return value
    return default


def _infer_outcome(attrs: dict[str, Any], spans: tuple[Span, ...]) -> str:
    for key in ("task.success", "evaluation.success", "success"):
        if key in attrs:
            return "success" if bool(attrs[key]) else "failure"
    for span in spans:
        for key in ("task.success", "evaluation.success", "success"):
            if key in span.attributes:
                return "success" if bool(span.attributes[key]) else "failure"
    return "failure" if any(span.failed for span in spans) else "success"


def load_otel_jsonl(path: str | Path) -> list[AgentTrace]:
    """Load newline-delimited OpenTelemetry-shaped traces."""

    traces: list[AgentTrace] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                traces.append(normalize_otel_trace(json.loads(stripped)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return traces


def save_otel_jsonl(traces: Iterable[AgentTrace], path: str | Path) -> None:
    """Save normalized traces in a simple OpenTelemetry-shaped JSONL form."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(_trace_to_raw(trace), sort_keys=True) + "\n")


def _trace_to_raw(trace: AgentTrace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "task_name": trace.task_name,
        "outcome": trace.outcome,
        "attributes": dict(trace.attributes),
        "spans": [
            {
                "name": span.name,
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "start_time": span.start_time,
                "end_time": span.end_time,
                "status_code": span.status_code,
                "attributes": dict(span.attributes),
                "events": list(span.events),
            }
            for span in trace.spans
        ],
    }


def load_public_exgentic(
    limit: int = 100,
    split: str = "train",
    fallback_path: str | Path | None = None,
) -> list[AgentTrace]:
    """Load public Exgentic OpenTelemetry-shaped traces when optional deps exist.

    Falls back to a local JSONL fixture when `datasets` or network access is not
    available. This keeps examples reproducible while preserving a path to public
    traces for research runs.
    """

    try:
        from datasets import load_dataset  # type: ignore

        rows = load_dataset("Exgentic/agent-llm-traces-v2", split=split)
        traces = []
        for idx, row in enumerate(rows):
            if idx >= limit:
                break
            traces.append(normalize_otel_trace(dict(row)))
        return traces
    except Exception:
        if fallback_path is None:
            raise
        return load_otel_jsonl(fallback_path)
