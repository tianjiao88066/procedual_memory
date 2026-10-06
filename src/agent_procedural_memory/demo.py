from __future__ import annotations

from pathlib import Path

from .evaluation import EvaluationSuite
from .exploration import TraceExplorer, WorkflowGraph, WorkflowNode
from .miner import ProceduralMemoryMiner
from .otel import load_otel_jsonl, save_otel_jsonl
from .toy_agent import RefundWorkflowAgent, build_refund_eval_cases


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = root / "data" / "public_trace_fixtures" / "support_refund_otel.jsonl"
    artifacts = root / "artifacts" / "refund_workflow"
    artifacts.mkdir(parents=True, exist_ok=True)

    traces = load_otel_jsonl(fixture)
    exploration = TraceExplorer().explore(_refund_workflow_graph(), "support_refund", _refund_outcome)
    save_otel_jsonl(exploration.traces, artifacts / "exploratory_traces.jsonl")

    skill = ProceduralMemoryMiner(min_success_support=0.5).mine_skill(traces, skill_name="support_refund_skill")
    skill_path = artifacts / "SKILL.md"
    skill_path.write_text(skill.to_markdown(), encoding="utf-8")

    success_sequences = tuple(trace.action_sequence for trace in traces if trace.succeeded)
    few_shot = success_sequences[:2]
    suite = EvaluationSuite(build_refund_eval_cases(), RefundWorkflowAgent())
    report = suite.compare(skill, few_shot_traces=few_shot)

    print(f"Loaded {len(traces)} OpenTelemetry-shaped traces from {fixture}")
    print(
        f"Generated {exploration.path_count} exploratory traces covering "
        f"{len(exploration.covered_actions)} workflow actions"
    )
    print(f"Generated procedural memory skill: {skill_path}")
    print()
    print(report.to_text())
    print()
    print("Representative generated skill procedure:")
    for step in skill.procedure:
        print(f"- {step.action}: {step.rationale}")


def _refund_workflow_graph() -> WorkflowGraph:
    return WorkflowGraph(
        start="authenticate_customer",
        terminal="notify_customer",
        nodes={
            "authenticate_customer": WorkflowNode("authenticate_customer", ("lookup_order", "auth_failed")),
            "auth_failed": WorkflowNode("auth_failed"),
            "lookup_order": WorkflowNode("lookup_order", ("request_receipt", "check_policy")),
            "request_receipt": WorkflowNode("request_receipt", ("check_policy",)),
            "check_policy": WorkflowNode("check_policy", ("issue_refund", "escalate_manual_review")),
            "escalate_manual_review": WorkflowNode("escalate_manual_review", ("issue_refund",)),
            "issue_refund": WorkflowNode("issue_refund", ("notify_customer",)),
            "notify_customer": WorkflowNode("notify_customer"),
        },
    )


def _refund_outcome(path: tuple[str, ...]) -> str:
    required = {"authenticate_customer", "lookup_order", "check_policy", "issue_refund", "notify_customer"}
    return "success" if required.issubset(path) and path[-1] == "notify_customer" else "failure"


if __name__ == "__main__":
    main()
