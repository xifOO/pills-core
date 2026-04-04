from typing import Dict

from pills_core._enums import TransformPhase
from pills_core.strategies.base import SingleStrategy

_PHASE = TransformPhase

_DEFAULT_PHASE_ORDER: Dict[_PHASE, int] = {
    _PHASE.IMPUTATION: 0,
    _PHASE.OUTLIER: 1,
    _PHASE.SCALING: 2,
}


def _phase_sort_key(phase_order: Dict[_PHASE, int], phase: _PHASE) -> tuple[int, str]:
    return (phase_order.get(phase, len(phase_order)), phase.value)


def _topological_sort(
    nodes: set[_PHASE],
    edges: set[tuple[_PHASE, _PHASE]],
    phase_order: Dict[_PHASE, int] = _DEFAULT_PHASE_ORDER,
) -> list[_PHASE]:
    """
    Perform a stable topological sort over TransformPhase nodes.

    Strategies contribute only hard constraints. If several phases are otherwise
    unconstrained, we fall back to the canonical preprocessing order defined in
    `_DEFAULT_PHASE_ORDER`.
    """
    from collections import defaultdict, deque

    in_degree: Dict[_PHASE, int] = dict.fromkeys(nodes, 0)
    adj: dict[_PHASE, list[_PHASE]] = defaultdict(list)

    for src, dst in edges:
        adj[src].append(dst)
        in_degree[dst] += 1

    queue = deque(
        sorted(
            (n for n in nodes if in_degree[n] == 0),
            key=lambda phase: _phase_sort_key(phase_order, phase),
        )
    )
    result: list[_PHASE] = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                queue = deque(
                    sorted(queue, key=lambda phase: _phase_sort_key(phase_order, phase))
                )

    if len(result) != len(nodes):
        raise RuntimeError(f"Cycle detected in phase graph. edges={edges}")

    return result


def resolve_phase_order(*strategies: SingleStrategy) -> list[_PHASE]:
    nodes = {s.phase for s in strategies}

    edges: set[tuple[_PHASE, _PHASE]] = {
        (src, dst)
        for s in strategies
        for src, dst in s.ordering_constraints(nodes)
        if src in nodes and dst in nodes
    }

    return _topological_sort(nodes, edges)
