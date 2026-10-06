from __future__ import annotations

from .models import EvaluationCase, RunResult, Skill


class RefundWorkflowAgent:
    """Deterministic toy agent used to demonstrate evaluation mechanics."""

    def run(
        self,
        case: EvaluationCase,
        skill: Skill | None = None,
        few_shot: tuple[tuple[str, ...], ...] = (),
    ) -> RunResult:
        if skill is not None:
            condition = "procedural_skill"
            actions = tuple(self._with_skill(case, skill))
        elif few_shot:
            condition = "few_shot_success_traces"
            actions = tuple(self._with_few_shot(case, few_shot))
        else:
            condition = "baseline"
            actions = tuple(self._baseline(case))
        return self._score(case, condition, actions)

    def _baseline(self, case: EvaluationCase) -> list[str]:
        actions = ["lookup_order", "issue_refund"]
        if case.attributes.get("needs_policy_check"):
            actions.insert(1, "check_policy")
        return actions

    def _with_few_shot(self, case: EvaluationCase, few_shot: tuple[tuple[str, ...], ...]) -> list[str]:
        if not few_shot:
            return self._baseline(case)
        actions = list(few_shot[0])
        if case.attributes.get("missing_receipt") and "request_receipt" not in actions:
            actions.insert(max(1, len(actions) - 2), "request_receipt")
        return actions

    def _with_skill(self, case: EvaluationCase, skill: Skill) -> list[str]:
        actions = list(skill.required_actions())
        if case.attributes.get("missing_receipt") and "request_receipt" not in actions:
            insert_at = actions.index("check_policy") if "check_policy" in actions else len(actions)
            actions.insert(insert_at, "request_receipt")
        if case.attributes.get("manual_review") and "escalate_manual_review" not in actions:
            insert_at = actions.index("issue_refund") if "issue_refund" in actions else len(actions)
            actions.insert(insert_at, "escalate_manual_review")
        return actions

    def _score(self, case: EvaluationCase, condition: str, actions: tuple[str, ...]) -> RunResult:
        expected = list(case.expected_actions)
        missing = [action for action in expected if action not in actions]
        extra_bad = [action for action in actions if action in case.distractor_actions]
        order_ok = _is_subsequence(expected, list(actions))
        success = not missing and not extra_bad and order_ok
        matched = sum(action in actions for action in expected)
        score = matched / len(expected)
        if extra_bad:
            score -= 0.25 * len(extra_bad)
        if not order_ok:
            score -= 0.25
        score = max(0.0, min(1.0, score))
        notes = tuple([*(f"missing:{action}" for action in missing), *(f"bad_extra:{action}" for action in extra_bad)])
        return RunResult(case.case_id, condition, actions, success, score, notes)


def build_refund_eval_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            "standard_refund",
            "support_refund",
            ("authenticate_customer", "lookup_order", "check_policy", "issue_refund", "notify_customer"),
            ("ship_replacement",),
            {"needs_policy_check": True},
        ),
        EvaluationCase(
            "missing_receipt",
            "support_refund",
            ("authenticate_customer", "lookup_order", "request_receipt", "check_policy", "issue_refund", "notify_customer"),
            ("issue_refund_without_receipt",),
            {"missing_receipt": True, "needs_policy_check": True},
        ),
        EvaluationCase(
            "manual_review",
            "support_refund",
            ("authenticate_customer", "lookup_order", "check_policy", "escalate_manual_review", "issue_refund", "notify_customer"),
            ("auto_approve_high_value_refund",),
            {"manual_review": True, "needs_policy_check": True},
        ),
    ]


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    cursor = 0
    for action in actual:
        if cursor < len(expected) and action == expected[cursor]:
            cursor += 1
    return cursor == len(expected)
