# support_refund_skill

Procedural memory for completing `support_refund` workflows. Mined from 3 successful and 3 failed trace(s).

## Triggers
- public_agent_workflow_fixture
- support_refund

## Prerequisites
- Use OpenTelemetry trace evidence before changing the procedure.

## Procedure
1. `authenticate_customer`: Common in successful workflows; execute it after prior required steps are verified. Evidence: 3 trace(s).
2. `lookup_order`: Common in successful workflows; execute it after prior required steps are verified. Evidence: 3 trace(s).
3. `check_policy`: Common in successful workflows; execute it after prior required steps are verified. Evidence: 3 trace(s).
4. `issue_refund`: Common in successful workflows; execute it after prior required steps are verified. Evidence: 3 trace(s).
5. `notify_customer`: Appears only in successful workflows and should be treated as a reliable step. Evidence: 3 trace(s).

## Failure Modes
- Trace `pub-demo-failure-001` failed after missing successful-step(s): authenticate_customer, check_policy, escalate_manual_review, notify_customer.
- Trace `pub-demo-failure-001` had `issue_refund` fail with `unauthenticated_customer`.
- Trace `pub-demo-failure-002` failed after missing successful-step(s): check_policy, escalate_manual_review, issue_refund, notify_customer.
- Trace `pub-demo-failure-002` had `issue_refund_without_receipt` fail with `missing_receipt`.
- Trace `pub-demo-failure-003` failed after missing successful-step(s): escalate_manual_review, issue_refund, notify_customer, request_receipt.
- Trace `pub-demo-failure-003` had `final_answer` fail with `retry_loop_no_terminal_refund`.
- Trace `pub-demo-failure-003` repeated action(s), suggesting a loop or retry trap: check_policy.

## Recovery Rules
- Before terminal response, compare completed actions against the required procedure and run any missing required step.
- When a tool span reports an error, inspect the error, repair inputs, and retry once before changing strategy.
- If the same action repeats, stop the loop, summarize current state, and resume from the next unmet required step.

## Verification
- Confirm the `support_refund` task outcome is successful before finalizing.
- Confirm required actions were completed in order: authenticate_customer, lookup_order, check_policy, issue_refund, notify_customer.
- Check the run against known failure modes and recovery rules.

## Metadata
- `failure_count`: 3
- `min_success_support`: 0.5
- `success_count`: 3
- `trace_count`: 6
