"""Bayesian confidence scoring for procedural memory steps.

Uses the Beta-Binomial model to estimate step reliability from trace evidence.
This approach is grounded in Bayesian statistics and applied in the procedural
learning literature (Anderson, 1993 — ACT-R; Tennenholtz et al., 2024 —
reinforcement learning from traces).

Key idea: each action step is modelled as a Bernoulli trial — it either
contributes to task success or it does not.  The Beta(α, β) posterior gives
a principled confidence interval that shrinks as evidence accumulates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StepConfidence:
    """Bayesian confidence estimate for a single procedure step."""

    action: str
    successes: int          # times this action appeared in successful traces
    failures: int           # times this action appeared in failed traces
    total_success_traces: int
    total_failure_traces: int

    # Beta prior: α₀=1, β₀=1 (uniform / uninformative prior)
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    @property
    def alpha(self) -> float:
        """Posterior α = prior_α + observed successes."""
        return self.prior_alpha + self.successes

    @property
    def beta(self) -> float:
        """Posterior β = prior_β + observed failures-with-action-present."""
        return self.prior_beta + self.failures

    @property
    def mean(self) -> float:
        """Posterior mean: E[θ] = α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Posterior variance."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def credible_interval(self, level: float = 0.90) -> tuple[float, float]:
        """Approximate symmetric credible interval using Normal approximation.

        For small sample sizes this is approximate; for production systems
        with many traces the Normal approximation to the Beta is adequate
        (as justified by the Bernstein-von Mises theorem).
        """
        z = _z_score(level)
        lower = max(0.0, self.mean - z * self.std)
        upper = min(1.0, self.mean + z * self.std)
        return (lower, upper)

    @property
    def is_reliable(self) -> bool:
        """Step is reliable if lower bound of 90% CI > 0.5."""
        lower, _ = self.credible_interval(0.90)
        return lower > 0.5

    @property
    def evidence_strength(self) -> str:
        """Qualitative assessment of evidence quantity."""
        n = self.successes + self.failures
        if n <= 2:
            return "weak"
        if n <= 10:
            return "moderate"
        return "strong"


def compute_step_confidence(
    action: str,
    success_traces: list,
    failure_traces: list,
) -> StepConfidence:
    """Compute Bayesian confidence for an action from trace lists."""
    successes = sum(
        1 for trace in success_traces
        if action in trace.action_sequence
    )
    failures = sum(
        1 for trace in failure_traces
        if action in trace.action_sequence
    )
    return StepConfidence(
        action=action,
        successes=successes,
        failures=failures,
        total_success_traces=len(success_traces),
        total_failure_traces=len(failure_traces),
    )


def _z_score(level: float) -> float:
    """Approximate z-score for common confidence levels."""
    table = {0.80: 1.282, 0.85: 1.440, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    if level in table:
        return table[level]
    # Rational approximation (Abramowitz & Stegun 26.2.23)
    p = (1.0 + level) / 2.0
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    )
