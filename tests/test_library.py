from __future__ import annotations

from pathlib import Path
import unittest

from agent_procedural_memory import (
    ConsolidationConfig,
    EvaluationSuite,
    LLMConfig,
    NullLLMBackend,
    ProceduralMemoryMiner,
    StepConfidence,
    TraceExplorer,
    WorkflowGraph,
    WorkflowNode,
    compute_step_confidence,
    consolidate_skills,
    load_otel_jsonl,
)
from agent_procedural_memory.confidence import _z_score
from agent_procedural_memory.consolidation import _similarity, _deterministic_merge
from agent_procedural_memory.toy_agent import RefundWorkflowAgent, build_refund_eval_cases


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "public_trace_fixtures" / "support_refund_otel.jsonl"


class MinerTests(unittest.TestCase):
    def test_mine_skill_from_success_and_failure_traces(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces, skill_name="support_refund_skill")

        self.assertEqual(skill.name, "support_refund_skill")
        self.assertIn("authenticate_customer", skill.required_actions())
        self.assertIn("notify_customer", skill.required_actions())
        self.assertTrue(skill.failure_modes)
        self.assertIn("Failure Modes", skill.to_markdown())

    def test_mine_skill_includes_confidence_metadata(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        # Confidence metadata should be present for each action
        confidence_keys = [k for k in skill.metadata if k.startswith("confidence.")]
        self.assertGreater(len(confidence_keys), 0)
        for key in confidence_keys:
            entry = skill.metadata[key]
            self.assertIn("mean", entry)
            self.assertIn("ci_lower", entry)
            self.assertIn("ci_upper", entry)
            self.assertIn("evidence_strength", entry)
            self.assertGreaterEqual(entry["mean"], 0.0)
            self.assertLessEqual(entry["mean"], 1.0)

    def test_mine_skill_deterministic_without_llm(self) -> None:
        """Miner with no LLM config produces identical results across runs."""
        traces = load_otel_jsonl(FIXTURE)
        miner = ProceduralMemoryMiner()
        skill_a = miner.mine_skill(traces, skill_name="test")
        skill_b = miner.mine_skill(traces, skill_name="test")
        self.assertEqual(
            [s.action for s in skill_a.procedure],
            [s.action for s in skill_b.procedure],
        )

    def test_mine_skill_with_null_llm_falls_back(self) -> None:
        """When LLM is configured but null, deterministic fallback is used."""
        traces = load_otel_jsonl(FIXTURE)
        llm_config = LLMConfig(backend=NullLLMBackend())
        miner = ProceduralMemoryMiner(llm_config=llm_config)
        skill = miner.mine_skill(traces)
        # Should still produce a valid skill
        self.assertTrue(skill.procedure)
        self.assertTrue(skill.description)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_compares_three_conditions(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        few_shot = tuple(trace.action_sequence for trace in traces if trace.succeeded)[:2]

        report = EvaluationSuite(build_refund_eval_cases(), RefundWorkflowAgent()).compare(skill, few_shot)

        self.assertEqual(
            set(report.condition_scores),
            {"baseline", "few_shot_success_traces", "procedural_skill"},
        )
        self.assertGreater(report.condition_scores["procedural_skill"], report.condition_scores["baseline"])
        self.assertGreaterEqual(
            report.condition_scores["procedural_skill"],
            report.condition_scores["few_shot_success_traces"],
        )

    def test_evaluation_with_bootstrap_ci(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        few_shot = tuple(trace.action_sequence for trace in traces if trace.succeeded)[:2]

        suite = EvaluationSuite(
            build_refund_eval_cases(),
            RefundWorkflowAgent(),
            bootstrap_samples=500,
            seed=42,
        )
        report = suite.compare(skill, few_shot)

        # Should have condition_metrics with CIs
        for condition, metrics in report.condition_metrics.items():
            self.assertIsNotNone(metrics.score_ci)
            lower, upper = metrics.score_ci
            self.assertLessEqual(lower, metrics.avg_score + 0.01)  # tolerance
            self.assertGreaterEqual(upper, metrics.avg_score - 0.01)

    def test_evaluation_report_text_includes_ci(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        suite = EvaluationSuite(
            build_refund_eval_cases(),
            RefundWorkflowAgent(),
            bootstrap_samples=100,
            seed=42,
        )
        report = suite.compare(skill)
        text = report.to_text()
        self.assertIn("95% CI:", text)


class ExplorationTests(unittest.TestCase):
    def test_trace_explorer_generates_path_covering_traces(self) -> None:
        graph = WorkflowGraph(
            start="start",
            terminal="done",
            nodes={
                "start": WorkflowNode("start", ("fast_path", "careful_path")),
                "fast_path": WorkflowNode("fast_path", ("done",)),
                "careful_path": WorkflowNode("careful_path", ("review",)),
                "review": WorkflowNode("review", ("done",)),
                "done": WorkflowNode("done"),
            },
        )

        report = TraceExplorer().explore(graph, "branching_task")

        self.assertEqual(report.path_count, 2)
        self.assertFalse(report.uncovered_actions)
        self.assertTrue(all(trace.task_name == "branching_task" for trace in report.traces))


class ConfidenceTests(unittest.TestCase):
    def test_step_confidence_basic(self) -> None:
        conf = StepConfidence(
            action="test_action",
            successes=8,
            failures=2,
            total_success_traces=10,
            total_failure_traces=5,
        )
        # With uniform prior: alpha=9, beta=3 → mean ≈ 0.75
        self.assertAlmostEqual(conf.mean, 9.0 / 12.0, places=3)
        self.assertGreater(conf.variance, 0)
        lower, upper = conf.credible_interval(0.90)
        self.assertLess(lower, conf.mean)
        self.assertGreater(upper, conf.mean)

    def test_step_confidence_all_success(self) -> None:
        conf = StepConfidence(
            action="reliable",
            successes=10,
            failures=0,
            total_success_traces=10,
            total_failure_traces=5,
        )
        self.assertGreater(conf.mean, 0.9)
        self.assertTrue(conf.is_reliable)

    def test_step_confidence_weak_evidence(self) -> None:
        conf = StepConfidence(
            action="rare",
            successes=1,
            failures=0,
            total_success_traces=1,
            total_failure_traces=0,
        )
        self.assertEqual(conf.evidence_strength, "weak")

    def test_compute_step_confidence_from_traces(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        successes = [t for t in traces if t.succeeded]
        failures = [t for t in traces if not t.succeeded]
        conf = compute_step_confidence("authenticate_customer", successes, failures)
        self.assertGreater(conf.successes, 0)
        self.assertGreater(conf.mean, 0.5)

    def test_z_score_known_values(self) -> None:
        self.assertAlmostEqual(_z_score(0.95), 1.960, places=2)
        self.assertAlmostEqual(_z_score(0.90), 1.645, places=2)


class ConsolidationTests(unittest.TestCase):
    def test_similarity_identical_skills(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        self.assertAlmostEqual(_similarity(skill, skill), 1.0)

    def test_deterministic_merge(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces, skill_name="a")
        skill_b = ProceduralMemoryMiner().mine_skill(traces, skill_name="b")
        merged = _deterministic_merge(skill, skill_b)
        self.assertIn("a", merged.name)
        self.assertIn("b", merged.name)
        # Merged should have at least as many steps as either original
        self.assertGreaterEqual(len(merged.procedure), len(skill.procedure))

    def test_consolidate_identical_skills_merges(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        miner = ProceduralMemoryMiner()
        skill_a = miner.mine_skill(traces, skill_name="a")
        skill_b = miner.mine_skill(traces, skill_name="b")
        result = consolidate_skills([skill_a, skill_b])
        # Should merge to one since they're identical
        self.assertEqual(len(result), 1)

    def test_consolidate_empty_list(self) -> None:
        self.assertEqual(consolidate_skills([]), [])

    def test_consolidate_single_skill_unchanged(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces)
        result = consolidate_skills([skill])
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
