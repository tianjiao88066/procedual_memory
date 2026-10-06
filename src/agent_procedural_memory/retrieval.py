"""Embedding-based skill retrieval with deterministic fallback.

Provides a ``SkillIndex`` that stores skills and retrieves the most relevant
ones for a given task context.  When an ``EmbeddingBackend`` is configured,
retrieval uses cosine similarity over embedding vectors.  Otherwise, it falls
back to Jaccard similarity over lowercased token sets — the same set-overlap
approach used in the consolidation module.

Academic grounding:
- Voyager (Wang et al., 2023) uses embedding-based retrieval over a skill
  library indexed by task description.
- CoALA (Sumers et al., 2024) identifies cue-based retrieval as the mechanism
  that selects which procedural memory to activate.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .llm import EmbeddingBackend, NullEmbeddingBackend
from .models import Skill

logger = logging.getLogger(__name__)


@dataclass
class SkillEntry:
    """An indexed skill with its text key and optional embedding vector."""

    skill: Skill
    text_key: str
    embedding: tuple[float, ...] | None = None
    tokens: frozenset[str] = field(default_factory=frozenset)


class SkillIndex:
    """Index and retrieve skills by task-context similarity.

    Parameters
    ----------
    embedding_backend : EmbeddingBackend | None
        When provided and not a ``NullEmbeddingBackend``, cosine similarity
        over embedding vectors is used.  Otherwise, Jaccard similarity over
        lowercased token sets provides a zero-dependency fallback.
    """

    def __init__(self, embedding_backend: EmbeddingBackend | None = None) -> None:
        self._backend = embedding_backend
        self._entries: list[SkillEntry] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(self, skill: Skill) -> None:
        """Index a single skill."""
        text_key = self._build_text_key(skill)
        tokens = _tokenize(text_key)
        embedding = self._embed_single(text_key)
        self._entries.append(SkillEntry(
            skill=skill,
            text_key=text_key,
            embedding=embedding,
            tokens=tokens,
        ))

    def add_many(self, skills: Iterable[Skill]) -> None:
        """Index multiple skills, batching embedding calls when possible."""
        skills_list = list(skills)
        if not skills_list:
            return
        text_keys = [self._build_text_key(s) for s in skills_list]
        token_sets = [_tokenize(tk) for tk in text_keys]
        embeddings = self._embed_batch(text_keys)
        for skill, text_key, tokens, emb in zip(skills_list, text_keys, token_sets, embeddings):
            self._entries.append(SkillEntry(
                skill=skill,
                text_key=text_key,
                embedding=emb,
                tokens=tokens,
            ))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Skill, float]]:
        """Return the top-k most relevant skills for *query*, with scores.

        Scores are in [0, 1] regardless of the similarity method used.
        """
        if not self._entries:
            return []

        if self._embeddings_available():
            return self._retrieve_by_embedding(query, top_k)
        return self._retrieve_by_jaccard(query, top_k)

    def _retrieve_by_embedding(self, query: str, top_k: int) -> list[tuple[Skill, float]]:
        query_emb = self._embed_single(query)
        if query_emb is None:
            # Embedding failed — fall back to Jaccard
            return self._retrieve_by_jaccard(query, top_k)

        scored: list[tuple[SkillEntry, float]] = []
        for entry in self._entries:
            if entry.embedding is None:
                continue
            sim = _cosine_similarity(query_emb, entry.embedding)
            scored.append((entry, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(entry.skill, score) for entry, score in scored[:top_k]]

    def _retrieve_by_jaccard(self, query: str, top_k: int) -> list[tuple[Skill, float]]:
        query_tokens = _tokenize(query)
        scored: list[tuple[SkillEntry, float]] = []
        for entry in self._entries:
            sim = _jaccard(query_tokens, entry.tokens)
            scored.append((entry, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(entry.skill, score) for entry, score in scored[:top_k]]

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, skill_name: str) -> bool:
        """Remove a skill by name. Returns True if found and removed."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.skill.name != skill_name]
        return len(self._entries) < before

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the index to a JSON file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [
                {
                    "skill_name": e.skill.name,
                    "text_key": e.text_key,
                    "embedding": list(e.embedding) if e.embedding else None,
                    "tokens": sorted(e.tokens),
                    "skill": _skill_to_dict(e.skill),
                }
                for e in self._entries
            ],
        }
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embedding_backend: EmbeddingBackend | None = None) -> SkillIndex:
        """Load an index from a JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(embedding_backend=embedding_backend)
        for entry_data in raw.get("entries", []):
            skill = _dict_to_skill(entry_data["skill"])
            emb = tuple(entry_data["embedding"]) if entry_data.get("embedding") else None
            tokens = frozenset(entry_data.get("tokens", []))
            index._entries.append(SkillEntry(
                skill=skill,
                text_key=entry_data["text_key"],
                embedding=emb,
                tokens=tokens,
            ))
        return index

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def skill_names(self) -> list[str]:
        return [e.skill.name for e in self._entries]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_text_key(self, skill: Skill) -> str:
        parts = [skill.name, skill.description]
        parts.extend(skill.triggers)
        parts.extend(step.action for step in skill.procedure if step.required)
        return " ".join(parts)

    def _embeddings_available(self) -> bool:
        return (
            self._backend is not None
            and not isinstance(self._backend, NullEmbeddingBackend)
        )

    def _embed_single(self, text: str) -> tuple[float, ...] | None:
        if not self._embeddings_available():
            return None
        assert self._backend is not None
        try:
            results = self._backend.embed([text])
            if results and results[0]:
                return tuple(results[0])
        except Exception as exc:
            logger.warning("Embedding call failed: %s", exc)
        return None

    def _embed_batch(self, texts: list[str]) -> list[tuple[float, ...] | None]:
        if not self._embeddings_available():
            return [None] * len(texts)
        assert self._backend is not None
        try:
            results = self._backend.embed(texts)
            return [tuple(v) if v else None for v in results]
        except Exception as exc:
            logger.warning("Batch embedding call failed: %s", exc)
            return [None] * len(texts)


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------

def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity between two vectors, clamped to [0, 1]."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokenize(text: str) -> frozenset[str]:
    """Split text into lowercased tokens, stripping punctuation."""
    return frozenset(
        word for word in (
            "".join(ch if ch.isalnum() else " " for ch in text).lower().split()
        )
        if len(word) > 1
    )


# ---------------------------------------------------------------------------
# Skill serialization helpers
# ---------------------------------------------------------------------------

def _skill_to_dict(skill: Skill) -> dict:
    from .models import SkillStep
    return {
        "name": skill.name,
        "description": skill.description,
        "triggers": skill.triggers,
        "prerequisites": skill.prerequisites,
        "procedure": [
            {
                "action": s.action,
                "rationale": s.rationale,
                "evidence_count": s.evidence_count,
                "required": s.required,
                "confidence": s.confidence,
                "confidence_interval": list(s.confidence_interval) if s.confidence_interval else None,
                "depends_on": list(s.depends_on),
            }
            for s in skill.procedure
        ],
        "failure_modes": skill.failure_modes,
        "recovery_rules": skill.recovery_rules,
        "verification": skill.verification,
        "metadata": skill.metadata,
        "version": skill.version,
    }


def _dict_to_skill(d: dict) -> Skill:
    from .models import SkillStep
    return Skill(
        name=d["name"],
        description=d["description"],
        triggers=d.get("triggers", []),
        prerequisites=d.get("prerequisites", []),
        procedure=[
            SkillStep(
                action=s["action"],
                rationale=s["rationale"],
                evidence_count=s.get("evidence_count", 0),
                required=s.get("required", True),
                confidence=s.get("confidence"),
                confidence_interval=tuple(s["confidence_interval"]) if s.get("confidence_interval") else None,
                depends_on=tuple(s.get("depends_on", ())),
            )
            for s in d.get("procedure", [])
        ],
        failure_modes=d.get("failure_modes", []),
        recovery_rules=d.get("recovery_rules", []),
        verification=d.get("verification", []),
        metadata=d.get("metadata", {}),
        version=d.get("version", 0),
    )
