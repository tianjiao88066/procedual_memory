from __future__ import annotations

import unittest

from agent_procedural_memory import (
    GeneralizationReport,
    SubProcedure,
    extract_sub_procedures,
)
from agent_procedural_memory.models import Skill, SkillStep


def _make_skill(name: str, task: str, actions: list[str]) -> Skill:
    return Skill(
        name=name,
        description=f"Skill for {task}",
        triggers=[task],
        procedure=[SkillStep(action=a, rationale="test", evidence_count=3) for a in actions],
    )


class GeneralizationTests(unittest.TestCase):
    def test_extract_common_subsequence_across_tasks(self) -> None:
        """Shared sub-procedure across different tasks is found."""
        refund = _make_skill("refund", "support_refund", ["authenticate", "lookup", "check_policy", "issue_refund", "notify"])
        exchange = _make_skill("exchange", "support_exchange", ["authenticate", "lookup", "check_policy", "issue_exchange", "notify"])

        report = extract_sub_procedures([refund, exchange], min_support=2, min_length=2)

        self.assertGreater(len(report.sub_procedures), 0)
        # "authenticate, lookup, check_policy" should be found
        found_actions = [sp.actions for sp in report.sub_procedures]
        self.assertTrue(
            any(("authenticate", "lookup", "check_policy") == sp or
                set(sp).issuperset({"authenticate", "lookup"})
                for sp in found_actions),
            f"Expected common prefix in {found_actions}",
        )

    def test_min_support_filters_rare_fragments(self) -> None:
        """Fragments in only 1 skill are excluded when min_support=2."""
        skill_a = _make_skill("a", "task_a", ["x", "y", "z"])
        skill_b = _make_skill("b", "task_b", ["p", "q", "r"])

        report = extract_sub_procedures([skill_a, skill_b], min_support=2)
        # No shared actions, so no sub-procedures
        self.assertEqual(len(report.sub_procedures), 0)

    def test_maximal_pruning(self) -> None:
        """A 2-gram substring of a frequent 3-gram is pruned."""
        skill_a = _make_skill("a", "task_a", ["auth", "lookup", "check", "do"])
        skill_b = _make_skill("b", "task_b", ["auth", "lookup", "check", "other"])

        report = extract_sub_procedures([skill_a, skill_b], min_support=2, min_length=2, max_length=4)

        # Should find "auth, lookup, check" as the maximal fragment
        # "auth, lookup" should be pruned (it's a substring)
        actions_list = [sp.actions for sp in report.sub_procedures]
        self.assertIn(("auth", "lookup", "check"), actions_list)
        # The 2-gram "auth, lookup" should be pruned
        self.assertNotIn(("auth", "lookup"), actions_list)

    def test_cross_task_only_excludes_same_task(self) -> None:
        """When cross_task_only=True, fragments only in same-task skills are excluded."""
        skill_a = _make_skill("a", "same_task", ["auth", "lookup", "do_a"])
        skill_b = _make_skill("b", "same_task", ["auth", "lookup", "do_b"])

        report = extract_sub_procedures([skill_a, skill_b], min_support=2, cross_task_only=True)
        # Same task name — no cross-task fragments
        self.assertEqual(len(report.sub_procedures), 0)

    def test_cross_task_false_includes_same_task(self) -> None:
        """When cross_task_only=False, same-task fragments are included."""
        skill_a = _make_skill("a", "same_task", ["auth", "lookup", "do_a"])
        skill_b = _make_skill("b", "same_task", ["auth", "lookup", "do_b"])

        report = extract_sub_procedures([skill_a, skill_b], min_support=2, cross_task_only=False)
        self.assertGreater(len(report.sub_procedures), 0)

    def test_no_skills_returns_empty(self) -> None:
        report = extract_sub_procedures([], min_support=2)
        self.assertEqual(len(report.sub_procedures), 0)

    def test_single_skill_returns_empty(self) -> None:
        skill = _make_skill("a", "task", ["auth", "lookup"])
        report = extract_sub_procedures([skill], min_support=2)
        self.assertEqual(len(report.sub_procedures), 0)

    def test_sub_procedure_to_skill_conversion(self) -> None:
        sp = SubProcedure(
            actions=("auth", "lookup", "check"),
            source_skills=("refund", "exchange"),
            support=2,
            description="Common auth-lookup-check fragment",
        )
        skill = sp.to_skill("auth_check_fragment")
        self.assertEqual(skill.name, "auth_check_fragment")
        self.assertEqual(len(skill.procedure), 3)
        self.assertEqual(skill.procedure[0].action, "auth")
        self.assertEqual(skill.metadata["kind"], "sub_procedure")

    def test_report_method_is_ngram_without_llm(self) -> None:
        skill_a = _make_skill("a", "task_a", ["auth", "lookup", "do"])
        skill_b = _make_skill("b", "task_b", ["auth", "lookup", "other"])
        report = extract_sub_procedures([skill_a, skill_b], min_support=2)
        self.assertEqual(report.method, "ngram")

    def test_report_source_skill_count(self) -> None:
        skills = [_make_skill(f"s{i}", f"task_{i}", ["auth", "lookup"]) for i in range(4)]
        report = extract_sub_procedures(skills, min_support=2)
        self.assertEqual(report.source_skill_count, 4)


if __name__ == "__main__":
    unittest.main()
