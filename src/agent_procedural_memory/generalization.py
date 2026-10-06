"""Cross-task generalization via frequent sub-procedure extraction.

Discovers transferable action fragments that recur across multiple skills
from different tasks.  The deterministic path uses contiguous n-gram
subsequence mining; the LLM-augmented path adds semantic alignment of
near-miss fragments.

Academic grounding:
- ExpeL (Zhao et al., 2023) extracts cross-task insights from trajectories.
- ExpeTrans (2025) transfers textual experiences between tasks.
- HTN planning (Erol et al., 1994) decomposes tasks into reusable sub-tasks.
- Frequent subsequence mining is a standard pattern mining technique
  (Agrawal & Srikant, 1995 — "Mining Sequential Patterns").
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .llm import SUBPROCEDURE_ALIGN_PROMPT, LLMConfig, llm_complete_json
from .models import Skill

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubProcedure:
    """A transferable action fragment found across multiple skills."""

    actions: tuple[str, ...]
    source_skills: tuple[str, ...]
    support: int
    description: str = ""

    def to_skill(self, name: str | None = None) -> Skill:
        """Convert to a Skill for indexing or reuse."""
        from .models import SkillStep
        skill_name = name or "_".join(self.actions[:3]) + "_fragment"
        return Skill(
            name=skill_name,
            description=self.description or f"Reusable sub-procedure found in {self.support} skill(s).",
            triggers=[f"sub-procedure from: {', '.join(self.source_skills[:5])}"],
            procedure=[
                SkillStep(action=action, rationale="Part of a cross-task sub-procedure.", evidence_count=self.support)
                for action in self.actions
            ],
            metadata={
                "source_skills": list(self.source_skills),
                "support": self.support,
                "kind": "sub_procedure",
            },
        )


@dataclass(frozen=True)
class GeneralizationReport:
    """Results of cross-task sub-procedure extraction."""

    sub_procedures: tuple[SubProcedure, ...]
    source_skill_count: int
    method: str  # "ngram" or "llm_aligned"


def extract_sub_procedures(
    skills: Sequence[Skill],
    *,
    min_support: int = 2,
    min_length: int = 2,
    max_length: int = 6,
    cross_task_only: bool = True,
    llm_config: LLMConfig | None = None,
) -> GeneralizationReport:
    """Extract frequent sub-procedures shared across skills.

    Parameters
    ----------
    skills : Sequence[Skill]
        The skill library to analyze.
    min_support : int
        Minimum number of distinct skills a fragment must appear in.
    min_length : int
        Minimum number of actions in a fragment.
    max_length : int
        Maximum number of actions in a fragment.
    cross_task_only : bool
        When True, only count fragments appearing in skills with different
        task names (extracted from ``skill.triggers[0]``).
    llm_config : LLMConfig | None
        When provided, semantically align near-miss fragments.
    """
    if len(skills) < min_support:
        return GeneralizationReport((), len(skills), "ngram")

    # Step 1: Extract action sequences from each skill
    skill_sequences: list[tuple[str, tuple[str, ...]]] = []
    for skill in skills:
        actions = tuple(step.action for step in skill.procedure if step.required)
        skill_sequences.append((skill.name, actions))

    # Step 2: Mine contiguous n-grams
    ngram_sources: dict[tuple[str, ...], set[str]] = {}
    for skill_name, actions in skill_sequences:
        for length in range(min_length, min(max_length + 1, len(actions) + 1)):
            for start in range(len(actions) - length + 1):
                ngram = actions[start : start + length]
                ngram_sources.setdefault(ngram, set()).add(skill_name)

    # Step 3: Filter by support
    frequent: dict[tuple[str, ...], set[str]] = {}
    for ngram, sources in ngram_sources.items():
        if len(sources) < min_support:
            continue
        if cross_task_only:
            # Check that sources come from skills with different task names
            task_names = set()
            for skill in skills:
                if skill.name in sources:
                    task_name = skill.triggers[0] if skill.triggers else skill.name
                    task_names.add(task_name)
            if len(task_names) < 2:
                continue
        frequent[ngram] = sources

    # Step 4: Maximal pruning — remove sub-sequences of longer frequent n-grams
    maximal = _maximal_filter(frequent)

    # Step 5: Build SubProcedure objects
    candidates = [
        SubProcedure(
            actions=ngram,
            source_skills=tuple(sorted(sources)),
            support=len(sources),
            description=f"Common sub-procedure: {' → '.join(ngram)}",
        )
        for ngram, sources in sorted(maximal.items(), key=lambda x: (-len(x[1]) * len(x[0]), x[0]))
    ]

    method = "ngram"

    # Step 6: Optional LLM alignment of near-miss fragments
    if llm_config and llm_config.is_available() and len(candidates) > 1:
        aligned = _llm_align(candidates, llm_config)
        if aligned is not None:
            candidates = aligned
            method = "llm_aligned"

    return GeneralizationReport(
        sub_procedures=tuple(candidates),
        source_skill_count=len(skills),
        method=method,
    )


def _maximal_filter(
    frequent: dict[tuple[str, ...], set[str]],
) -> dict[tuple[str, ...], set[str]]:
    """Remove n-grams that are contiguous sub-sequences of a longer frequent n-gram."""
    to_remove: set[tuple[str, ...]] = set()
    sorted_by_length = sorted(frequent.keys(), key=len, reverse=True)
    for i, longer in enumerate(sorted_by_length):
        for shorter in sorted_by_length[i + 1 :]:
            if shorter in to_remove:
                continue
            if _is_contiguous_sub(shorter, longer) and frequent[shorter] <= frequent[longer]:
                to_remove.add(shorter)
    return {k: v for k, v in frequent.items() if k not in to_remove}


def _is_contiguous_sub(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    """Check if *short* is a contiguous sub-sequence of *long*."""
    n = len(short)
    return any(long[i : i + n] == short for i in range(len(long) - n + 1))


def _llm_align(
    candidates: list[SubProcedure],
    llm_config: LLMConfig,
) -> list[SubProcedure] | None:
    """Try to merge near-miss candidates using LLM semantic judgment."""
    if len(candidates) < 2:
        return None

    merged: list[SubProcedure] = list(candidates)
    merged_any = False

    i = 0
    while i < len(merged):
        j = i + 1
        while j < len(merged):
            a, b = merged[i], merged[j]
            # Only attempt alignment on pairs with some overlap
            set_a = set(a.actions)
            set_b = set(b.actions)
            if not set_a & set_b and _jaccard_actions(set_a, set_b) < 0.2:
                j += 1
                continue

            prompt = SUBPROCEDURE_ALIGN_PROMPT.format(
                seq_a=" → ".join(a.actions),
                skills_a=", ".join(a.source_skills[:3]),
                seq_b=" → ".join(b.actions),
                skills_b=", ".join(b.source_skills[:3]),
            )
            result = llm_complete_json(llm_config, prompt)
            if result and result.get("same"):
                # Merge: keep the longer one, union sources
                keeper = a if len(a.actions) >= len(b.actions) else b
                new_sources = tuple(sorted(set(a.source_skills + b.source_skills)))
                desc = result.get("description", keeper.description)
                merged_sp = SubProcedure(
                    actions=keeper.actions,
                    source_skills=new_sources,
                    support=len(new_sources),
                    description=desc,
                )
                merged[i] = merged_sp
                merged.pop(j)
                merged_any = True
                continue  # Re-check j position
            j += 1
        i += 1

    return merged if merged_any else None


def _jaccard_actions(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
