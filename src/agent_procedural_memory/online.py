"""Online learning — incrementally refine skills from new trace evidence.

Provides an ``OnlineLearner`` that maintains running statistics and
updates an existing ``Skill`` one trace at a time.  The core update is
O(number of actions) per trace because the Beta-Binomial conjugate
model's posterior update is a single increment to α or β.

Academic grounding:
- ACT-R (Anderson, 1993): production rule strength increases with
  successful practice and decays with disuse — our confidence updates
  mirror this directly.
- Reflexion (Shinn et al., 2023): self-improvement loop where the
  agent incorporates task feedback — our ``update()`` is the
  non-verbal equivalent, updating statistics rather than text.
- Online Bayesian updating: the Beta-Binomial is the textbook
  conjugate model for incremental learning from binary outcomes
  (Bernardo & Smith, 2000).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .confidence import StepConfidence, compute_step_confidence
from .llm import REFINEMENT_PROMPT, LLMConfig, llm_complete_json
from .models import AgentTrace, Skill, SkillStep

logger = logging.getLogger(__name__)


@dataclass
class OnlineLearner:
    """Incrementally refine an existing Skill from new trace evidence.

    Parameters
    ----------
    skill : Skill
        The current skill being refined.
    action_success_counts : dict[str, int]
        Per-action count of appearances in successful traces.
    action_failure_counts : dict[str, int]
        Per-action count of appearances in failed traces.
    total_success_traces : int
        Total successful traces incorporated.
    total_failure_traces : int
        Total failed traces incorporated.
    min_success_support : float
        Fraction threshold for promoting an action to required.
    confidence_level : float
        Credible interval level for Bayesian scoring.
    demotion_threshold : float
        Lower CI bound below which a required step is demoted to optional.
    llm_config : LLMConfig | None
        Optional LLM for re-synthesizing rationales and descriptions.
    """

    skill: Skill
    action_success_counts: dict[str, int] = field(default_factory=dict)
    action_failure_counts: dict[str, int] = field(default_factory=dict)
    total_success_traces: int = 0
    total_failure_traces: int = 0
    min_success_support: float = 0.5
    confidence_level: float = 0.90
    demotion_threshold: float = 0.3
    llm_config: LLMConfig | None = None

    # Track cumulative changes for LLM refinement
    _pending_confidence_deltas: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_skill(cls, skill: Skill, llm_config: LLMConfig | None = None) -> OnlineLearner:
        """Bootstrap an OnlineLearner from an existing Skill's metadata."""
        success_counts: dict[str, int] = {}
        failure_counts: dict[str, int] = {}

        # Extract per-action counts from confidence metadata
        for key, value in skill.metadata.items():
            if key.startswith("confidence.") and isinstance(value, dict):
                action = key[len("confidence."):]
                # Approximate counts from mean and evidence strength
                # If we have the original StepConfidence data, it's more precise
                mean = value.get("mean", 0.5)
                strength = value.get("evidence_strength", "weak")
                # Estimate: for Beta(α, β) with mean=α/(α+β), we need total n
                # We stored evidence_count in the SkillStep
                step = next((s for s in skill.procedure if s.action == action), None)
                if step:
                    n_success = step.evidence_count
                    success_counts[action] = n_success
                    # Estimate failure count from confidence mean
                    if mean > 0 and mean < 1:
                        n_fail = max(0, round(n_success * (1 - mean) / mean))
                        failure_counts[action] = n_fail

        total_success = skill.metadata.get("success_count", sum(success_counts.values()) // max(1, len(success_counts)))
        total_failure = skill.metadata.get("failure_count", 0)

        return cls(
            skill=skill,
            action_success_counts=success_counts,
            action_failure_counts=failure_counts,
            total_success_traces=total_success,
            total_failure_traces=total_failure,
            min_success_support=skill.metadata.get("min_success_support", 0.5),
            llm_config=llm_config,
        )

    def update(self, trace: AgentTrace) -> Skill:
        """Incorporate one new trace and return the refined Skill."""
        actions_in_trace = set(trace.action_sequence)

        # Step 1: Update running counts
        if trace.succeeded:
            self.total_success_traces += 1
            for action in actions_in_trace:
                old = self.action_success_counts.get(action, 0)
                self.action_success_counts[action] = old + 1
        else:
            self.total_failure_traces += 1
            for action in actions_in_trace:
                old = self.action_failure_counts.get(action, 0)
                self.action_failure_counts[action] = old + 1

        # Step 2: Recompute confidence for all known actions
        all_actions = set(self.action_success_counts) | set(self.action_failure_counts)
        for action in self.skill.required_actions():
            all_actions.add(action)

        confidences: dict[str, StepConfidence] = {}
        for action in all_actions:
            s = self.action_success_counts.get(action, 0)
            f = self.action_failure_counts.get(action, 0)
            confidences[action] = StepConfidence(
                action=action,
                successes=s,
                failures=f,
                total_success_traces=self.total_success_traces,
                total_failure_traces=self.total_failure_traces,
            )

        # Step 3: Build updated procedure
        existing_actions = {step.action for step in self.skill.procedure}
        support_threshold = max(1, round(self.total_success_traces * self.min_success_support))
        updated_procedure = self._update_procedure(confidences, existing_actions, support_threshold)

        # Step 4: Update failure modes if trace failed
        updated_failure_modes = list(self.skill.failure_modes)
        if not trace.succeeded:
            new_modes = self._extract_failure_modes(trace, existing_actions | set(a for a, _ in updated_procedure))
            for mode in new_modes:
                if mode not in updated_failure_modes:
                    updated_failure_modes.append(mode)
            # Cap at 12
            updated_failure_modes = updated_failure_modes[:12]

        # Step 5: Track confidence deltas for LLM refinement
        for action, conf in confidences.items():
            old_mean = self._pending_confidence_deltas.get(action)
            if old_mean is None:
                # First time — find the old confidence from the skill
                old_step = next((s for s in self.skill.procedure if s.action == action), None)
                old_mean = old_step.confidence if old_step and old_step.confidence is not None else 0.5
            delta = abs(conf.mean - old_mean)
            if delta > 0.1:
                self._pending_confidence_deltas[action] = conf.mean

        # Step 6: Build new SkillStep list
        new_steps: list[SkillStep] = []
        for action, required in updated_procedure:
            conf = confidences.get(action)
            ci = conf.credible_interval(self.confidence_level) if conf else None
            rationale = self._get_rationale(action, conf)
            new_steps.append(SkillStep(
                action=action,
                rationale=rationale,
                evidence_count=self.action_success_counts.get(action, 0),
                required=required,
                confidence=round(conf.mean, 3) if conf else None,
                confidence_interval=(round(ci[0], 3), round(ci[1], 3)) if ci else None,
            ))

        # Step 7: Optionally refine description via LLM
        description = self._maybe_refine_description(new_steps)

        # Step 8: Build new Skill
        new_version = self.skill.version + 1
        updated_metadata = dict(self.skill.metadata)
        updated_metadata["trace_count"] = self.total_success_traces + self.total_failure_traces
        updated_metadata["success_count"] = self.total_success_traces
        updated_metadata["failure_count"] = self.total_failure_traces
        for action, conf in confidences.items():
            ci_low, ci_high = conf.credible_interval(self.confidence_level)
            updated_metadata[f"confidence.{action}"] = {
                "mean": round(conf.mean, 3),
                "ci_lower": round(ci_low, 3),
                "ci_upper": round(ci_high, 3),
                "evidence_strength": conf.evidence_strength,
            }

        new_skill = Skill(
            name=self.skill.name,
            description=description,
            triggers=self.skill.triggers,
            prerequisites=self.skill.prerequisites,
            procedure=new_steps,
            failure_modes=updated_failure_modes,
            recovery_rules=self.skill.recovery_rules,
            verification=self.skill.verification,
            metadata=updated_metadata,
            version=new_version,
        )
        self.skill = new_skill
        return new_skill

    def update_batch(self, traces: Iterable[AgentTrace]) -> Skill:
        """Incorporate multiple traces sequentially."""
        skill = self.skill
        for trace in traces:
            skill = self.update(trace)
        return skill

    # ------------------------------------------------------------------
    # Internal: procedure update logic
    # ------------------------------------------------------------------

    def _update_procedure(
        self,
        confidences: dict[str, StepConfidence],
        existing_actions: set[str],
        support_threshold: int,
    ) -> list[tuple[str, bool]]:
        """Return updated (action, required) pairs preserving order."""
        result: list[tuple[str, bool]] = []

        # Keep existing steps, updating required status
        for step in self.skill.procedure:
            conf = confidences.get(step.action)
            if conf:
                ci_low, _ = conf.credible_interval(self.confidence_level)
                required = self.action_success_counts.get(step.action, 0) >= support_threshold
                # Demote if confidence has dropped
                if ci_low < self.demotion_threshold and step.required:
                    required = False
                    logger.info("Demoting step '%s' (CI lower=%.3f < %.3f)", step.action, ci_low, self.demotion_threshold)
            else:
                required = step.required
            result.append((step.action, required))

        # Promote new actions that cross the support threshold
        for action, conf in confidences.items():
            if action in existing_actions:
                continue
            s_count = self.action_success_counts.get(action, 0)
            if s_count >= support_threshold:
                logger.info("Promoting new action '%s' to procedure (support=%d)", action, s_count)
                result.append((action, True))

        return result

    # ------------------------------------------------------------------
    # Internal: failure mode extraction
    # ------------------------------------------------------------------

    def _extract_failure_modes(self, trace: AgentTrace, known_actions: set[str]) -> list[str]:
        modes: list[str] = []
        sequence = trace.action_sequence
        missing = [a for a in known_actions if a not in sequence]
        failed_spans = [span for span in trace.spans if span.failed]
        repeated = [a for a, count in Counter(sequence).items() if count > 1]

        if missing:
            modes.append(
                f"Trace `{trace.trace_id}` failed after missing step(s): "
                f"{', '.join(sorted(missing)[:4])}."
            )
        for span in failed_spans[:2]:
            reason = span.attributes.get("error.type") or span.attributes.get("exception.type") or span.status_code
            modes.append(f"Trace `{trace.trace_id}` had `{span.action_name}` fail with `{reason}`.")
        if repeated:
            modes.append(
                f"Trace `{trace.trace_id}` repeated action(s): {', '.join(sorted(repeated))}."
            )
        return modes

    # ------------------------------------------------------------------
    # Internal: rationale generation
    # ------------------------------------------------------------------

    def _get_rationale(self, action: str, conf: StepConfidence | None) -> str:
        """Generate a rationale for an action step."""
        # Try to keep existing rationale if confidence hasn't changed much
        existing = next((s for s in self.skill.procedure if s.action == action), None)
        if existing and action not in self._pending_confidence_deltas:
            return existing.rationale

        s = self.action_success_counts.get(action, 0)
        f = self.action_failure_counts.get(action, 0)
        conf_suffix = ""
        if conf:
            ci_low, ci_high = conf.credible_interval(self.confidence_level)
            conf_suffix = f" [confidence: {conf.mean:.2f}, {self.confidence_level:.0%} CI: ({ci_low:.2f}, {ci_high:.2f})]"

        if s and not f:
            return f"Appears only in successful workflows.{conf_suffix}"
        if s >= f:
            return f"Common in successful workflows; execute after prior steps verified.{conf_suffix}"
        return f"Useful but risky; execute only when preconditions are satisfied.{conf_suffix}"

    # ------------------------------------------------------------------
    # Internal: LLM-augmented refinement
    # ------------------------------------------------------------------

    def _maybe_refine_description(self, new_steps: list[SkillStep]) -> str:
        """Optionally use LLM to refine description when significant changes occurred."""
        if (
            not self._pending_confidence_deltas
            or not self.llm_config
            or not self.llm_config.is_available()
        ):
            return self.skill.description

        changes_text = "\n".join(
            f"- {action}: confidence changed to {new_mean:.3f}"
            for action, new_mean in self._pending_confidence_deltas.items()
        )
        prompt = REFINEMENT_PROMPT.format(
            changes=changes_text,
            description=self.skill.description,
        )
        result = llm_complete_json(self.llm_config, prompt)
        if result and result.get("description"):
            # Also update rationales for changed steps if LLM provided them
            rationales = result.get("rationales", {})
            if rationales:
                for i, step in enumerate(new_steps):
                    if step.action in rationales:
                        new_steps[i] = SkillStep(
                            action=step.action,
                            rationale=rationales[step.action],
                            evidence_count=step.evidence_count,
                            required=step.required,
                            confidence=step.confidence,
                            confidence_interval=step.confidence_interval,
                            depends_on=step.depends_on,
                        )
            self._pending_confidence_deltas.clear()
            return result["description"]
        return self.skill.description

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize learner state for persistence."""
        from .retrieval import _skill_to_dict
        return {
            "skill": _skill_to_dict(self.skill),
            "action_success_counts": dict(self.action_success_counts),
            "action_failure_counts": dict(self.action_failure_counts),
            "total_success_traces": self.total_success_traces,
            "total_failure_traces": self.total_failure_traces,
            "min_success_support": self.min_success_support,
            "confidence_level": self.confidence_level,
            "demotion_threshold": self.demotion_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], llm_config: LLMConfig | None = None) -> OnlineLearner:
        """Restore learner from serialized state."""
        from .retrieval import _dict_to_skill
        return cls(
            skill=_dict_to_skill(data["skill"]),
            action_success_counts=data.get("action_success_counts", {}),
            action_failure_counts=data.get("action_failure_counts", {}),
            total_success_traces=data.get("total_success_traces", 0),
            total_failure_traces=data.get("total_failure_traces", 0),
            min_success_support=data.get("min_success_support", 0.5),
            confidence_level=data.get("confidence_level", 0.90),
            demotion_threshold=data.get("demotion_threshold", 0.3),
            llm_config=llm_config,
        )

    def save(self, path: str | Path) -> None:
        """Persist learner state to a JSON file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, llm_config: LLMConfig | None = None) -> OnlineLearner:
        """Load learner from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, llm_config=llm_config)
