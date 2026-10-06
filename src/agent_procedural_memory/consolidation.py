"""Skill consolidation — merge and abstract related procedural skills.

Implements both deterministic and LLM-augmented consolidation strategies:

1. **Jaccard overlap** (deterministic): Merge skills whose required action
   sets overlap above a configurable threshold.  This is a standard set-
   similarity measure used in information retrieval (Manning et al., 2008).

2. **LLM-judged semantic merge** (non-deterministic): When an LLM backend is
   available, ask it to judge whether two skills are semantically equivalent
   even when action names differ (e.g. ``verify_identity`` vs
   ``authenticate_customer``).

The consolidation loop follows the agglomerative clustering pattern:
repeatedly merge the most similar pair until no pair exceeds the threshold.
This is analogous to hierarchical skill abstraction in HTN planning
(Erol et al., 1994).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from .llm import CONSOLIDATION_PROMPT, LLMConfig, llm_complete_json
from .models import Skill, SkillStep

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """Parameters controlling skill consolidation."""

    jaccard_threshold: float = 0.5
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    use_llm_judgment: bool = True
    max_merges: int = 10


def consolidate_skills(skills: Sequence[Skill], config: ConsolidationConfig | None = None) -> list[Skill]:
    """Merge overlapping skills into a reduced set.

    Returns a new list where highly overlapping skills have been merged.
    The original Skill objects are not mutated.
    """
    if not skills:
        return []
    cfg = config or ConsolidationConfig()
    pool = list(skills)
    merges_done = 0

    while len(pool) > 1 and merges_done < cfg.max_merges:
        best_i, best_j, best_score = -1, -1, 0.0
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                score = _similarity(pool[i], pool[j])
                if score > best_score:
                    best_i, best_j, best_score = i, j, score

        if best_score < cfg.jaccard_threshold:
            # Try LLM judgment for the top pair if deterministic overlap is low
            if cfg.use_llm_judgment and cfg.llm_config.is_available() and best_score > 0.1:
                merged = _llm_merge(pool[best_i], pool[best_j], cfg.llm_config)
                if merged is not None:
                    pool = [s for idx, s in enumerate(pool) if idx not in (best_i, best_j)]
                    pool.append(merged)
                    merges_done += 1
                    logger.info("LLM-merged '%s' + '%s' → '%s'", pool[best_i].name if best_i < len(pool) else "?", pool[best_j].name if best_j < len(pool) else "?", merged.name)
                    continue
            break

        merged = _deterministic_merge(pool[best_i], pool[best_j])
        logger.info(
            "Deterministic merge: '%s' + '%s' → '%s' (Jaccard=%.2f)",
            pool[best_i].name, pool[best_j].name, merged.name, best_score,
        )
        pool = [s for idx, s in enumerate(pool) if idx not in (best_i, best_j)]
        pool.append(merged)
        merges_done += 1

    return pool


def _similarity(a: Skill, b: Skill) -> float:
    """Jaccard similarity over required action sets."""
    set_a = set(a.required_actions())
    set_b = set(b.required_actions())
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _deterministic_merge(a: Skill, b: Skill) -> Skill:
    """Merge two skills by unioning their procedures and failure modes."""
    # Union steps, keeping higher evidence counts
    step_map: dict[str, SkillStep] = {}
    for step in (*a.procedure, *b.procedure):
        existing = step_map.get(step.action)
        if existing is None or step.evidence_count > existing.evidence_count:
            step_map[step.action] = step
    procedure = sorted(step_map.values(), key=lambda s: (-s.evidence_count, s.action))

    return Skill(
        name=f"{a.name}+{b.name}",
        description=f"Consolidated skill from '{a.name}' and '{b.name}'.",
        triggers=sorted(set(a.triggers + b.triggers)),
        prerequisites=sorted(set(a.prerequisites + b.prerequisites)),
        procedure=list(procedure),
        failure_modes=sorted(set(a.failure_modes + b.failure_modes)),
        recovery_rules=sorted(set(a.recovery_rules + b.recovery_rules)),
        verification=sorted(set(a.verification + b.verification)),
        metadata={
            "merged_from": [a.name, b.name],
            "trace_count": a.metadata.get("trace_count", 0) + b.metadata.get("trace_count", 0),
            "success_count": a.metadata.get("success_count", 0) + b.metadata.get("success_count", 0),
            "failure_count": a.metadata.get("failure_count", 0) + b.metadata.get("failure_count", 0),
        },
    )


def _llm_merge(a: Skill, b: Skill, llm_config: LLMConfig) -> Skill | None:
    """Ask the LLM whether two skills should merge; return merged skill or None."""
    prompt = CONSOLIDATION_PROMPT.format(
        skill_a_name=a.name,
        skill_a_steps=", ".join(s.action for s in a.procedure),
        skill_a_failures="; ".join(a.failure_modes[:3]) or "none",
        skill_b_name=b.name,
        skill_b_steps=", ".join(s.action for s in b.procedure),
        skill_b_failures="; ".join(b.failure_modes[:3]) or "none",
    )
    result = llm_complete_json(llm_config, prompt)
    if not result or not result.get("merge"):
        return None

    merged = _deterministic_merge(a, b)
    merged.name = result.get("merged_name", merged.name)
    merged.metadata["llm_merge_reason"] = result.get("reason", "")
    return merged
