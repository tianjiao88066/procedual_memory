from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Span:
    """Normalized representation of an OpenTelemetry span."""

    name: str
    span_id: str | None = None
    parent_span_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    status_code: str = "OK"
    attributes: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()

    @property
    def failed(self) -> bool:
        status = str(self.status_code).upper()
        attrs = self.attributes
        return (
            status in {"ERROR", "STATUS_CODE_ERROR"}
            or bool(attrs.get("error"))
            or bool(attrs.get("error.type"))
            or bool(attrs.get("exception.type"))
        )

    @property
    def action_name(self) -> str:
        attrs = self.attributes
        for key in (
            "agent.action",
            "tool.name",
            "gen_ai.tool.name",
            "gen_ai.operation.name",
            "operation.name",
        ):
            value = attrs.get(key)
            if value:
                return str(value)
        return self.name

    @property
    def kind(self) -> str:
        attrs = self.attributes
        for key in ("span.kind", "gen_ai.operation.name", "agent.step.kind"):
            value = attrs.get(key)
            if value:
                return str(value)
        if "tool.name" in attrs:
            return "tool"
        if "gen_ai.request.model" in attrs or "gen_ai.response.model" in attrs:
            return "llm"
        return "step"


@dataclass(frozen=True)
class AgentTrace:
    """Normalized agent trace with workflow metadata and ordered spans."""

    trace_id: str
    task_name: str
    outcome: str
    spans: tuple[Span, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.outcome == "success"

    @property
    def action_sequence(self) -> tuple[str, ...]:
        return tuple(span.action_name for span in self.spans if span.kind != "internal")


@dataclass(frozen=True)
class SkillStep:
    """One procedural instruction mined from traces.

    The ``confidence`` and ``confidence_interval`` fields carry Bayesian
    posterior estimates when the miner is configured to compute them.
    """

    action: str
    rationale: str
    evidence_count: int = 0
    required: bool = True
    confidence: float | None = None
    confidence_interval: tuple[float, float] | None = None
    depends_on: tuple[str, ...] = ()


@dataclass
class Skill:
    """Portable agent skill generated from trace evidence."""

    name: str
    description: str
    triggers: list[str]
    prerequisites: list[str] = field(default_factory=list)
    procedure: list[SkillStep] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    recovery_rules: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 0

    def required_actions(self) -> list[str]:
        return [step.action for step in self.procedure if step.required]

    def to_markdown(self) -> str:
        lines = [
            f"# {self.name}",
            "",
            self.description,
            "",
            "## Triggers",
        ]
        lines.extend(f"- {item}" for item in self.triggers)
        lines.extend(["", "## Prerequisites"])
        lines.extend(f"- {item}" for item in self.prerequisites or ["None."])
        lines.extend(["", "## Procedure"])
        for idx, step in enumerate(self.procedure, start=1):
            optional = " Optional." if not step.required else ""
            conf_str = ""
            if step.confidence is not None:
                conf_str = f" Confidence: {step.confidence:.2f}"
                if step.confidence_interval:
                    conf_str += f" CI: ({step.confidence_interval[0]:.2f}, {step.confidence_interval[1]:.2f})"
                conf_str += "."
            deps_str = ""
            if step.depends_on:
                deps_str = f" Depends on: {', '.join(step.depends_on)}."
            lines.append(
                f"{idx}. `{step.action}`: {step.rationale}"
                f" Evidence: {step.evidence_count} trace(s).{conf_str}{deps_str}{optional}"
            )
        lines.extend(["", "## Failure Modes"])
        lines.extend(f"- {item}" for item in self.failure_modes or ["No recurrent failure modes found."])
        lines.extend(["", "## Recovery Rules"])
        lines.extend(f"- {item}" for item in self.recovery_rules or ["Retry from the last verified safe step."])
        lines.extend(["", "## Verification"])
        lines.extend(f"- {item}" for item in self.verification or ["Confirm the terminal task outcome is successful."])
        if self.metadata:
            lines.extend(["", "## Metadata"])
            for key, value in sorted(self.metadata.items()):
                lines.append(f"- `{key}`: {value}")
        lines.append("")
        return "\n".join(lines)


@dataclass(frozen=True)
class EvaluationCase:
    """A deterministic task case for comparing agent memory conditions."""

    case_id: str
    task_name: str
    expected_actions: tuple[str, ...]
    distractor_actions: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    case_id: str
    condition: str
    actions: tuple[str, ...]
    success: bool
    score: float
    notes: tuple[str, ...] = ()
