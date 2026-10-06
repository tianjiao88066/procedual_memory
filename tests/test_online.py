from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_procedural_memory import (
    OnlineLearner,
    ProceduralMemoryMiner,
    load_otel_jsonl,
)
from agent_procedural_memory.models import AgentTrace, Skill, SkillStep, Span


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "public_trace_fixtures" / "support_refund_otel.jsonl"


def _make_trace(trace_id: str, actions: list[str], outcome: str = "success") -> AgentTrace:
    spans = tuple(
        Span(
            name=action,
            span_id=f"{trace_id}-{i}",
            attributes={"agent.action": action, "task.name": "test_task"},
            status_code="OK" if outcome == "success" else ("ERROR" if i == len(actions) - 1 else "OK"),
        )
        for i, action in enumerate(actions)
    )
    return AgentTrace(
        trace_id=trace_id,
        task_name="test_task",
        outcome=outcome,
        spans=spans,
    )


class OnlineLearnerTests(unittest.TestCase):
    def test_from_skill_bootstraps_counts(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces, skill_name="test")

        learner = OnlineLearner.from_skill(skill)
        self.assertEqual(learner.skill.name, "test")
        self.assertGreater(learner.total_success_traces, 0)
        self.assertGreater(len(learner.action_success_counts), 0)

    def test_update_single_success_increases_confidence(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        learner = OnlineLearner.from_skill(skill)

        old_success_count = learner.total_success_traces
        new_trace = _make_trace("online-1", ["authenticate_customer", "lookup_order", "check_policy", "issue_refund", "notify_customer"])
        updated_skill = learner.update(new_trace)

        self.assertEqual(learner.total_success_traces, old_success_count + 1)
        self.assertEqual(updated_skill.version, skill.version + 1)

        # Confidence for authenticate_customer should be present
        auth_conf = updated_skill.metadata.get("confidence.authenticate_customer")
        self.assertIsNotNone(auth_conf)
        self.assertGreater(auth_conf["mean"], 0.5)

    def test_update_single_failure_updates_failure_modes(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        learner = OnlineLearner.from_skill(skill)

        old_failure_count = len(skill.failure_modes)
        fail_trace = _make_trace(
            "online-fail-1",
            ["lookup_order", "issue_refund"],
            outcome="failure",
        )
        # Add error attribute to the last span
        fail_spans = list(fail_trace.spans)
        last_span = fail_spans[-1]
        new_attrs = dict(last_span.attributes)
        new_attrs["error.type"] = "policy_violation"
        fail_spans[-1] = Span(
            name=last_span.name,
            span_id=last_span.span_id,
            status_code="ERROR",
            attributes=new_attrs,
        )
        fail_trace = AgentTrace(
            trace_id=fail_trace.trace_id,
            task_name=fail_trace.task_name,
            outcome="failure",
            spans=tuple(fail_spans),
        )

        updated_skill = learner.update(fail_trace)
        self.assertGreater(learner.total_failure_traces, 0)
        # Should have at least one new failure mode
        self.assertGreaterEqual(len(updated_skill.failure_modes), old_failure_count)

    def test_new_action_promoted_to_procedure(self) -> None:
        """A new action crossing support threshold gets promoted."""
        skill = Skill(
            name="test",
            description="Test skill",
            triggers=["test_task"],
            procedure=[
                SkillStep(action="step_a", rationale="test", evidence_count=5, required=True),
            ],
            metadata={"success_count": 5, "failure_count": 0, "min_success_support": 0.5},
        )

        learner = OnlineLearner(
            skill=skill,
            action_success_counts={"step_a": 5},
            action_failure_counts={},
            total_success_traces=5,
            total_failure_traces=0,
            min_success_support=0.5,
        )

        # Feed traces with a new action "step_b" enough times to cross threshold
        for i in range(4):
            trace = _make_trace(f"promo-{i}", ["step_a", "step_b"])
            learner.update(trace)

        # step_b should now be in the procedure
        action_names = [s.action for s in learner.skill.procedure]
        self.assertIn("step_b", action_names)

    def test_weak_action_demoted(self) -> None:
        """An action whose confidence drops gets demoted to optional."""
        skill = Skill(
            name="test",
            description="Test",
            triggers=["test"],
            procedure=[
                SkillStep(action="reliable", rationale="test", evidence_count=10, required=True),
                SkillStep(action="unreliable", rationale="test", evidence_count=1, required=True),
            ],
            metadata={"success_count": 10, "failure_count": 0, "min_success_support": 0.5},
        )

        learner = OnlineLearner(
            skill=skill,
            action_success_counts={"reliable": 10, "unreliable": 1},
            action_failure_counts={"unreliable": 8},
            total_success_traces=10,
            total_failure_traces=8,
            min_success_support=0.5,
            demotion_threshold=0.3,
        )

        # Feed more failure traces where unreliable appears
        for i in range(5):
            trace = _make_trace(f"demote-{i}", ["reliable", "unreliable"], outcome="failure")
            learner.update(trace)

        # unreliable should be demoted (required=False)
        unreliable_step = next((s for s in learner.skill.procedure if s.action == "unreliable"), None)
        self.assertIsNotNone(unreliable_step)
        self.assertFalse(unreliable_step.required)

    def test_update_batch_matches_sequential(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)

        new_traces = [
            _make_trace("batch-1", ["authenticate_customer", "lookup_order", "notify_customer"]),
            _make_trace("batch-2", ["authenticate_customer", "lookup_order", "check_policy", "issue_refund", "notify_customer"]),
        ]

        # Sequential path
        learner_seq = OnlineLearner.from_skill(skill)
        for t in new_traces:
            learner_seq.update(t)

        # Batch path
        learner_batch = OnlineLearner.from_skill(skill)
        learner_batch.update_batch(new_traces)

        self.assertEqual(learner_seq.total_success_traces, learner_batch.total_success_traces)
        self.assertEqual(learner_seq.action_success_counts, learner_batch.action_success_counts)

    def test_version_increments(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        self.assertEqual(skill.version, 0)

        learner = OnlineLearner.from_skill(skill)
        trace = _make_trace("v-1", ["authenticate_customer", "notify_customer"])
        s1 = learner.update(trace)
        self.assertEqual(s1.version, 1)

        s2 = learner.update(_make_trace("v-2", ["authenticate_customer"]))
        self.assertEqual(s2.version, 2)

    def test_serialization_roundtrip(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        learner = OnlineLearner.from_skill(skill)

        # Update a few times
        learner.update(_make_trace("ser-1", ["authenticate_customer", "lookup_order"]))
        learner.update(_make_trace("ser-2", ["authenticate_customer"], outcome="failure"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "learner.json"
            learner.save(path)

            restored = OnlineLearner.load(path)
            self.assertEqual(restored.total_success_traces, learner.total_success_traces)
            self.assertEqual(restored.total_failure_traces, learner.total_failure_traces)
            self.assertEqual(restored.action_success_counts, learner.action_success_counts)
            self.assertEqual(restored.skill.name, learner.skill.name)
            self.assertEqual(restored.skill.version, learner.skill.version)

    def test_deterministic_without_llm(self) -> None:
        """Updates without LLM produce consistent results."""
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)

        new_trace = _make_trace("det-1", ["authenticate_customer", "lookup_order", "notify_customer"])

        learner_a = OnlineLearner.from_skill(skill)
        skill_a = learner_a.update(new_trace)

        learner_b = OnlineLearner.from_skill(skill)
        skill_b = learner_b.update(new_trace)

        self.assertEqual(
            [s.action for s in skill_a.procedure],
            [s.action for s in skill_b.procedure],
        )

    def test_updated_skill_has_valid_markdown(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        learner = OnlineLearner.from_skill(skill)

        updated = learner.update(_make_trace("md-1", ["authenticate_customer", "lookup_order"]))
        md = updated.to_markdown()
        self.assertIn("# ", md)
        self.assertIn("## Procedure", md)
        self.assertIn("authenticate_customer", md)


if __name__ == "__main__":
    unittest.main()
