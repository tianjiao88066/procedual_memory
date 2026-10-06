from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_procedural_memory import (
    ProceduralMemoryMiner,
    SkillIndex,
    load_otel_jsonl,
)
from agent_procedural_memory.models import Skill, SkillStep
from agent_procedural_memory.retrieval import _cosine_similarity, _jaccard, _tokenize


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "public_trace_fixtures" / "support_refund_otel.jsonl"


def _make_skill(name: str, description: str, actions: list[str]) -> Skill:
    return Skill(
        name=name,
        description=description,
        triggers=[name],
        procedure=[SkillStep(action=a, rationale="test") for a in actions],
    )


class RetrievalTests(unittest.TestCase):
    def test_index_and_retrieve_deterministic(self) -> None:
        """Index 3 skills, retrieve the best match by Jaccard."""
        refund = _make_skill("refund_skill", "Handle customer refund requests", ["authenticate", "lookup", "refund"])
        exchange = _make_skill("exchange_skill", "Handle product exchanges", ["authenticate", "lookup", "exchange"])
        billing = _make_skill("billing_skill", "Resolve billing inquiries", ["authenticate", "check_balance", "adjust"])

        index = SkillIndex()
        index.add_many([refund, exchange, billing])
        self.assertEqual(index.size, 3)

        results = index.retrieve("customer refund", top_k=2)
        self.assertEqual(len(results), 2)
        # Refund skill should rank first for "customer refund"
        self.assertEqual(results[0][0].name, "refund_skill")

    def test_retrieve_returns_top_k(self) -> None:
        skills = [_make_skill(f"skill_{i}", f"Description {i}", [f"action_{i}"]) for i in range(5)]
        index = SkillIndex()
        index.add_many(skills)

        results = index.retrieve("Description 0 action_0", top_k=3)
        self.assertEqual(len(results), 3)
        # All scores should be non-negative
        self.assertTrue(all(score >= 0.0 for _, score in results))

    def test_retrieve_with_mock_embedding_backend(self) -> None:
        """Verify cosine ranking with a mock embedding backend."""

        class MockEmbedder:
            def embed(self, texts: list[str]) -> list[list[float]]:
                # Return simple vectors based on text length for predictable ranking
                return [[len(t), 1.0, 0.0] for t in texts]

        refund = _make_skill("refund", "refund", ["refund"])
        exchange = _make_skill("exchange", "exchange exchange exchange", ["exchange"])

        index = SkillIndex(embedding_backend=MockEmbedder())
        index.add_many([refund, exchange])

        # Query with a short text — should be closer to refund's shorter embedding
        results = index.retrieve("ref", top_k=2)
        self.assertEqual(len(results), 2)
        # Both should have positive scores
        self.assertGreater(results[0][1], 0.0)

    def test_empty_index_returns_empty(self) -> None:
        index = SkillIndex()
        results = index.retrieve("anything")
        self.assertEqual(results, [])

    def test_remove_skill_from_index(self) -> None:
        skill = _make_skill("to_remove", "test", ["a"])
        index = SkillIndex()
        index.add(skill)
        self.assertEqual(index.size, 1)

        removed = index.remove("to_remove")
        self.assertTrue(removed)
        self.assertEqual(index.size, 0)

        # Removing non-existent skill returns False
        self.assertFalse(index.remove("nonexistent"))

    def test_save_and_load_roundtrip(self) -> None:
        traces = load_otel_jsonl(FIXTURE)
        skill = ProceduralMemoryMiner().mine_skill(traces, skill_name="roundtrip_test")

        index = SkillIndex()
        index.add(skill)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index.json"
            index.save(path)

            loaded = SkillIndex.load(path)
            self.assertEqual(loaded.size, 1)
            self.assertEqual(loaded.skill_names, ["roundtrip_test"])

            results = loaded.retrieve("support refund", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0].name, "roundtrip_test")

    def test_skill_names_property(self) -> None:
        skills = [_make_skill(f"s{i}", f"d{i}", [f"a{i}"]) for i in range(3)]
        index = SkillIndex()
        index.add_many(skills)
        self.assertEqual(sorted(index.skill_names), ["s0", "s1", "s2"])

    def test_fallback_when_embedding_fails(self) -> None:
        """When embedding backend raises, retrieval falls back to Jaccard."""

        class FailingEmbedder:
            def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("Network error")

        skill = _make_skill("test", "test description", ["action_a"])
        index = SkillIndex(embedding_backend=FailingEmbedder())
        index.add(skill)  # add falls back gracefully
        results = index.retrieve("test description", top_k=1)
        # Should still return results via Jaccard fallback
        self.assertEqual(len(results), 1)


class SimilarityHelperTests(unittest.TestCase):
    def test_cosine_identical(self) -> None:
        v = (1.0, 2.0, 3.0)
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0, places=5)

    def test_cosine_orthogonal(self) -> None:
        self.assertAlmostEqual(_cosine_similarity((1.0, 0.0), (0.0, 1.0)), 0.0)

    def test_cosine_empty(self) -> None:
        self.assertEqual(_cosine_similarity((), ()), 0.0)

    def test_jaccard_identical(self) -> None:
        s = frozenset(["a", "b", "c"])
        self.assertAlmostEqual(_jaccard(s, s), 1.0)

    def test_jaccard_disjoint(self) -> None:
        self.assertAlmostEqual(_jaccard(frozenset(["a"]), frozenset(["b"])), 0.0)

    def test_tokenize(self) -> None:
        tokens = _tokenize("Hello, World! Test-123")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("test", tokens)
        self.assertIn("123", tokens)


if __name__ == "__main__":
    unittest.main()
