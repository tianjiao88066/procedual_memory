"""LLM integration layer for non-deterministic procedural memory operations.

Provides an abstract LLM interface and prompt templates for:
- Rationale generation from trace evidence
- Causal dependency inference between procedure steps
- Semantic action clustering (grouping synonymous actions)
- Skill synthesis and summarization
- Skill consolidation across workflows

The design follows the «tool-augmented LLM» pattern described by Schick et al.
(2023, "Toolformer") — the library's deterministic algorithms remain the
backbone, and the LLM is injected only where human-like judgment adds value
that statistics alone cannot provide.

Also provides an abstract embedding interface for vector-based skill retrieval.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract LLM interface
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface that any LLM provider must satisfy.

    Implementations may wrap OpenAI, Anthropic, local models, or test stubs.
    The single ``complete`` method keeps the surface tiny and mockable.
    """

    def complete(self, prompt: str, *, temperature: float = 0.4, max_tokens: int = 1024) -> str:
        """Return a text completion for *prompt*."""
        ...


# ---------------------------------------------------------------------------
# Null / passthrough backend (deterministic fallback)
# ---------------------------------------------------------------------------

class NullLLMBackend:
    """Returns an empty string — used when no LLM is configured.

    Callers should treat an empty response as "LLM unavailable" and fall
    back to the deterministic path.
    """

    def complete(self, prompt: str, *, temperature: float = 0.4, max_tokens: int = 1024) -> str:
        return ""


# ---------------------------------------------------------------------------
# Abstract embedding interface
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingBackend(Protocol):
    """Minimal interface for embedding providers.

    Implementations may wrap OpenAI, Anthropic, sentence-transformers, etc.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""
        ...


class NullEmbeddingBackend:
    """Returns empty lists — triggers deterministic fallback in retrieval."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RATIONALE_PROMPT = """\
You are an expert at analysing agent execution traces.

Given the following evidence about an action step in a workflow, write a
concise one-sentence rationale explaining *why* this step matters and *when*
it should be executed relative to the other steps.

Action: {action}
Appears in {success_count}/{total_success} successful traces.
Appears in {failure_count}/{total_failure} failed traces.
Average position in successful traces: {avg_position:.1f}
Actions that typically precede it: {predecessors}
Actions that typically follow it: {successors}

Rationale:"""

CAUSAL_PROMPT = """\
You are an expert at analysing agent execution traces for causal dependencies.

Given the following procedure steps extracted from successful agent traces,
identify which steps causally depend on which. A step B depends on step A if
A must complete before B can succeed — not just because A happens to appear
earlier, but because B's precondition is established by A.

Steps (in canonical order):
{steps}

Respond with a JSON array of objects, each with "step", "depends_on" (list of
step names), and "reason" (one sentence). Only include genuine causal
dependencies, not mere ordering conventions.

JSON:"""

CLUSTER_PROMPT = """\
You are an expert at normalizing agent action names across different traces
and benchmarks.

Group the following action names into semantic clusters where each cluster
represents the same underlying capability. Actions that are clearly distinct
should each be their own cluster.

Actions:
{actions}

Respond with a JSON object mapping a canonical name (lowercase_snake_case) to
the list of original action names it represents. Only merge actions that are
truly synonymous.

JSON:"""

SYNTHESIS_PROMPT = """\
You are an expert at writing clear, actionable procedure documentation for AI
agents.

Given the following structured skill information mined from execution traces,
write a concise description (2-3 sentences) and a list of human-readable
trigger conditions.

Skill name: {name}
Task name: {task_name}
Number of procedure steps: {step_count}
Required steps: {required_steps}
Failure modes observed: {failure_modes}
Success rate in training traces: {success_rate:.0%}

Description (2-3 sentences):"""

CONSOLIDATION_PROMPT = """\
You are an expert at identifying when two agent skills describe overlapping
or equivalent procedures.

Skill A: {skill_a_name}
  Procedure: {skill_a_steps}
  Failure modes: {skill_a_failures}

Skill B: {skill_b_name}
  Procedure: {skill_b_steps}
  Failure modes: {skill_b_failures}

Are these skills overlapping enough to merge into a single skill?
If yes, respond with a JSON object:
{{"merge": true, "merged_name": "...", "reason": "...", "keep_steps_from": "A|B|both"}}
If no, respond with:
{{"merge": false, "reason": "..."}}

JSON:"""

JUDGE_PROMPT = """\
You are evaluating an AI agent's execution of a task. Score the execution
on a scale from 0.0 to 1.0.

Task: {task_name}
Expected actions: {expected}
Actual actions: {actual}
Distractor actions to avoid: {distractors}

Consider:
1. Were all required actions performed?
2. Were they in the correct order?
3. Were any harmful/distractor actions taken?
4. Was the approach efficient (no unnecessary steps)?

Respond with a JSON object: {{"score": <float>, "reasoning": "..."}}

JSON:"""

SUBPROCEDURE_ALIGN_PROMPT = """\
You are an expert at identifying when two action subsequences represent the
same underlying capability, even if the individual action names differ.

Subsequence A: {seq_a}
Found in skills: {skills_a}

Subsequence B: {seq_b}
Found in skills: {skills_b}

Are these the same sub-procedure (same purpose, same causal role in the
workflow)?

Respond with a JSON object:
{{"same": true/false, "canonical_name": "...", "description": "one sentence"}}

JSON:"""

REFINEMENT_PROMPT = """\
You are an expert at updating agent procedure documentation based on new
evidence.

A procedural skill is being refined with new trace evidence. The following
steps have had significant confidence changes:

{changes}

Current skill description: {description}

Write:
1. An updated description (2-3 sentences) reflecting the new evidence.
2. For each changed step, a revised one-sentence rationale.

Respond with a JSON object:
{{"description": "...", "rationales": {{"step_name": "rationale", ...}}}}

JSON:"""


# ---------------------------------------------------------------------------
# LLM call helpers with fallback
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configuration for LLM-augmented operations."""

    backend: LLMBackend = field(default_factory=NullLLMBackend)
    temperature: float = 0.4
    max_retries: int = 2
    fallback_on_error: bool = True

    def is_available(self) -> bool:
        """Return True if a real LLM backend is configured."""
        return not isinstance(self.backend, NullLLMBackend)


def llm_complete(config: LLMConfig, prompt: str, *, temperature: float | None = None) -> str:
    """Call the LLM with retry logic and graceful fallback."""
    temp = temperature if temperature is not None else config.temperature
    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            result = config.backend.complete(prompt, temperature=temp)
            if result:
                return result
            logger.debug("LLM returned empty response on attempt %d", attempt)
        except Exception as exc:
            last_error = exc
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, config.max_retries, exc)
    if last_error and not config.fallback_on_error:
        raise last_error
    return ""


def llm_complete_json(config: LLMConfig, prompt: str, *, temperature: float | None = None) -> Any:
    """Call the LLM and parse the response as JSON, returning None on failure."""
    raw = llm_complete(config, prompt, temperature=temperature)
    if not raw:
        return None
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON: %.200s", text)
        return None
