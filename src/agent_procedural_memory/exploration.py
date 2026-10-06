from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .models import AgentTrace, Span


@dataclass(frozen=True)
class WorkflowNode:
    """A workflow action and its possible next actions."""

    name: str
    next_actions: tuple[str, ...] = ()
    can_fail: bool = False


@dataclass
class WorkflowGraph:
    """Simple directed workflow graph for path-complete exploration."""

    start: str
    terminal: str
    nodes: dict[str, WorkflowNode]

    def validate(self) -> None:
        if self.start not in self.nodes:
            raise ValueError(f"Missing start node: {self.start}")
        if self.terminal not in self.nodes:
            raise ValueError(f"Missing terminal node: {self.terminal}")
        for node in self.nodes.values():
            for target in node.next_actions:
                if target not in self.nodes:
                    raise ValueError(f"Node {node.name} points to unknown target {target}")


@dataclass(frozen=True)
class ExplorationReport:
    traces: tuple[AgentTrace, ...]
    path_count: int
    covered_actions: tuple[str, ...]
    uncovered_actions: tuple[str, ...]


class TraceExplorer:
    """Generate traces that systematically cover workflow branches."""

    def __init__(self, max_depth: int = 12) -> None:
        self.max_depth = max_depth

    def explore(
        self,
        graph: WorkflowGraph,
        task_name: str,
        outcome_fn: Callable[[tuple[str, ...]], str] | None = None,
    ) -> ExplorationReport:
        """Enumerate acyclic paths and convert them into OTel-shaped traces."""

        graph.validate()
        paths = tuple(self._walk(graph, graph.start, (), set()))
        traces = tuple(
            self._path_to_trace(idx, task_name, path, outcome_fn)
            for idx, path in enumerate(paths, start=1)
        )
        all_actions = set(graph.nodes)
        covered = set(action for path in paths for action in path)
        return ExplorationReport(
            traces=traces,
            path_count=len(paths),
            covered_actions=tuple(sorted(covered)),
            uncovered_actions=tuple(sorted(all_actions - covered)),
        )

    def _walk(
        self,
        graph: WorkflowGraph,
        current: str,
        prefix: tuple[str, ...],
        seen: set[str],
    ) -> Iterable[tuple[str, ...]]:
        if len(prefix) >= self.max_depth:
            yield prefix + (current,)
            return
        node = graph.nodes[current]
        path = prefix + (current,)
        if current == graph.terminal or not node.next_actions:
            yield path
            return
        for target in node.next_actions:
            if target in seen:
                yield path + (target,)
            else:
                yield from self._walk(graph, target, path, seen | {current})

    def _path_to_trace(
        self,
        idx: int,
        task_name: str,
        path: tuple[str, ...],
        outcome_fn: Callable[[tuple[str, ...]], str] | None,
    ) -> AgentTrace:
        outcome = outcome_fn(path) if outcome_fn else ("success" if path[-1] in {"done", "success"} else "failure")
        spans = []
        for pos, action in enumerate(path, start=1):
            failed = outcome == "failure" and pos == len(path)
            spans.append(
                Span(
                    name=action,
                    span_id=f"{idx}-{pos}",
                    parent_span_id=f"{idx}-{pos - 1}" if pos > 1 else None,
                    status_code="ERROR" if failed else "OK",
                    attributes={
                        "agent.action": action,
                        "agent.step.index": pos,
                        "task.name": task_name,
                        "task.outcome": outcome,
                    },
                )
            )
        return AgentTrace(
            trace_id=f"explore-{idx}",
            task_name=task_name,
            outcome=outcome,
            spans=tuple(spans),
            attributes={"source": "trace_explorer", "task.name": task_name},
        )
