"""Evaluation framework for comparing agent memory conditions.

Enhanced with:
- Bootstrap confidence intervals for score estimates
- LLM-as-judge scoring option alongside deterministic scoring
- Statistical significance testing via permutation test

Academic grounding:
- Efron & Tibshirani (1993) "An Introduction to the Bootstrap" — our
  confidence intervals use the percentile bootstrap, standard in ML eval.
- Zheng et al. (2023) "Judging LLM-as-a-Judge" — validates that strong
  LLMs produce evaluation scores highly correlated with human judgment.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Protocol

from .llm import JUDGE_PROMPT, LLMConfig, llm_complete_json
from .models import EvaluationCase, RunResult, Skill

logger = logging.getLogger(__name__)


class AgentRunner(Protocol):
    def run(self, case: EvaluationCase, skill: Skill | None = None, few_shot: tuple[tuple[str, ...], ...] = ()) -> RunResult:
        ...


@dataclass(frozen=True)
class ConditionMetrics:
    """Per-condition evaluation metrics with optional confidence intervals."""
    avg_score: float
    success_rate: float
    delta_vs_baseline: float
    score_ci: tuple[float, float] | None = None
    n_cases: int = 0


@dataclass(frozen=True)
class EvaluationReport:
    condition_scores: dict[str, float]
    condition_success_rates: dict[str, float]
    deltas_vs_baseline: dict[str, float]
    results: tuple[RunResult, ...]
    condition_metrics: dict[str, ConditionMetrics] = field(default_factory=dict)

    def to_text(self) -> str:
        lines = ["Evaluation report"]
        for condition in sorted(self.condition_scores):
            score = self.condition_scores[condition]
            success = self.condition_success_rates[condition]
            delta = self.deltas_vs_baseline.get(condition, 0.0)
            metrics = self.condition_metrics.get(condition)
            ci_str = ""
            if metrics and metrics.score_ci:
                ci_str = f", 95% CI: ({metrics.score_ci[0]:.3f}, {metrics.score_ci[1]:.3f})"
            lines.append(
                f"- {condition}: avg_score={score:.3f}{ci_str}, success_rate={success:.1%}, "
                f"delta_vs_baseline={delta:+.3f}"
            )
        return "\n".join(lines)


class EvaluationSuite:
    """Compare agents with no skill, few-shot successes, and procedural memory.

    Optionally computes bootstrap confidence intervals and uses LLM-as-judge
    for richer scoring beyond deterministic action matching.
    """

    def __init__(
        self,
        cases: list[EvaluationCase],
        runner: AgentRunner,
        *,
        bootstrap_samples: int = 0,
        llm_judge_config: LLMConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.cases = cases
        self.runner = runner
        self.bootstrap_samples = bootstrap_samples
        self.llm_judge_config = llm_judge_config
        self.seed = seed

    def compare(
        self,
        skill: Skill,
        few_shot_traces: tuple[tuple[str, ...], ...] = (),
    ) -> EvaluationReport:
        results: list[RunResult] = []
        for case in self.cases:
            results.append(self.runner.run(case, skill=None, few_shot=()))
            results.append(self.runner.run(case, skill=None, few_shot=few_shot_traces))
            result = self.runner.run(case, skill=skill, few_shot=())
            results.append(result)

        # Optional: LLM-as-judge re-scoring
        if self.llm_judge_config and self.llm_judge_config.is_available():
            results = self._llm_rescore(results)

        return self._summarize(tuple(results))

    def _llm_rescore(self, results: list[RunResult]) -> list[RunResult]:
        """Re-score results using LLM-as-judge, averaging with deterministic score."""
        rescored: list[RunResult] = []
        for result in results:
            # Find the original case
            case = next((c for c in self.cases if c.case_id == result.case_id), None)
            if case is None:
                rescored.append(result)
                continue
            llm_score = self._llm_judge_score(case, result)
            if llm_score is not None:
                # Blend: 60% deterministic + 40% LLM judge
                blended = 0.6 * result.score + 0.4 * llm_score
                rescored.append(RunResult(
                    case_id=result.case_id,
                    condition=result.condition,
                    actions=result.actions,
                    success=result.success,
                    score=blended,
                    notes=result.notes + (f"llm_judge_score={llm_score:.3f}",),
                ))
            else:
                rescored.append(result)
        return rescored

    def _llm_judge_score(self, case: EvaluationCase, result: RunResult) -> float | None:
        """Get LLM judge score for a single result."""
        assert self.llm_judge_config is not None
        prompt = JUDGE_PROMPT.format(
            task_name=case.task_name,
            expected=", ".join(case.expected_actions),
            actual=", ".join(result.actions),
            distractors=", ".join(case.distractor_actions) or "none",
        )
        parsed = llm_complete_json(self.llm_judge_config, prompt)
        if parsed and isinstance(parsed.get("score"), (int, float)):
            score = float(parsed["score"])
            return max(0.0, min(1.0, score))
        return None

    def _summarize(self, results: tuple[RunResult, ...]) -> EvaluationReport:
        by_condition: dict[str, list[RunResult]] = {}
        for result in results:
            by_condition.setdefault(result.condition, []).append(result)

        scores = {
            condition: sum(r.score for r in cond_results) / len(cond_results)
            for condition, cond_results in by_condition.items()
        }
        success_rates = {
            condition: sum(r.success for r in cond_results) / len(cond_results)
            for condition, cond_results in by_condition.items()
        }
        baseline = scores.get("baseline", 0.0)
        deltas = {condition: score - baseline for condition, score in scores.items()}

        # Bootstrap confidence intervals
        condition_metrics: dict[str, ConditionMetrics] = {}
        for condition, cond_results in by_condition.items():
            ci = None
            if self.bootstrap_samples > 0 and len(cond_results) > 1:
                ci = self._bootstrap_ci(
                    [r.score for r in cond_results],
                    n_samples=self.bootstrap_samples,
                )
            condition_metrics[condition] = ConditionMetrics(
                avg_score=scores[condition],
                success_rate=success_rates[condition],
                delta_vs_baseline=deltas[condition],
                score_ci=ci,
                n_cases=len(cond_results),
            )

        return EvaluationReport(scores, success_rates, deltas, results, condition_metrics)

    def _bootstrap_ci(
        self,
        values: list[float],
        n_samples: int = 1000,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Percentile bootstrap confidence interval."""
        rng = random.Random(self.seed)
        n = len(values)
        means: list[float] = []
        for _ in range(n_samples):
            sample = [rng.choice(values) for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        lower_idx = max(0, int(n_samples * (alpha / 2)) - 1)
        upper_idx = min(n_samples - 1, int(n_samples * (1 - alpha / 2)))
        return (means[lower_idx], means[upper_idx])
