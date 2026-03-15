from typing import Dict, List, Set, Tuple

from pills_core._enums import TransformPhase
from pills_core.strategies.numeric.imputation import NumericalImputationStrategy
from pills_core.strategies.numeric.outliers import NumericalOutlierStrategy
from pills_core.strategies.numeric.scaling import NumericalScalingStrategy

_PHASE = TransformPhase


def _topological_sort(
    nodes: Set[_PHASE], edges: Set[Tuple[_PHASE, _PHASE]]
) -> List[_PHASE]:
    """
    Perform a topological sort over a set of TransformPhase nodes using Kahn's algorithm.

    A topological sort produces a linear ordering of nodes such that for every
    directed edge (A → B), node A appears before node B in the result. This is
    exactly what we need to determine a valid pipeline execution order where some
    phases must precede others.

    Kahn's algorithm works by repeatedly picking nodes with no incoming edges
    (i.e. no unresolved dependencies), appending them to the result, and removing
    their outgoing edges from the graph. If all nodes are consumed, the sort
    succeeded. If any nodes remain, a cycle exists — which means the constraint
    graph is contradictory and no valid ordering is possible.

    In this codebase a cycle should never occur in practice because `resolve_phase_order`
    resolves all conflicts before building the edge set. The cycle check is kept
    as a safety net to catch logic errors during development.
    """
    from collections import defaultdict, deque

    # in_degree tracks how many unresolved dependencies each node still has.
    # A node is only eligible for processing once its in_degree reaches 0,
    # meaning all phases that must precede it have already been scheduled.
    in_degree: Dict[_PHASE, int] = dict.fromkeys(nodes, 0)

    # adj is the adjacency list: adj[A] = [B, C, ...] means A must come before B and C.
    # defaultdict(list) is used so we can safely append without checking key existence.
    adj: dict[_PHASE, list[_PHASE]] = defaultdict(list)

    for src, dst in edges:
        adj[src].append(dst)
        in_degree[dst] += 1

    # Seed the queue with all nodes that have no dependencies at all.
    # deque is used instead of a list for O(1) popleft performance.
    queue = deque(n for n in nodes if in_degree[n] == 0)
    result: list[_PHASE] = []

    while queue:
        node = queue.popleft()
        result.append(node)

        # "Remove" all edges going out of this node by decrementing the
        # in_degree of its neighbors. If a neighbor's in_degree hits 0,
        # all its dependencies are satisfied and it becomes schedulable.
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(nodes):
        raise RuntimeError(f"Cycle detected in phase graph. edges={edges}")

    return result


def resolve_phase_order(
    imputation: NumericalImputationStrategy,
    outlier: NumericalOutlierStrategy,
    scaling: NumericalScalingStrategy,
) -> list[_PHASE]:
    """
    Derive the correct execution order for the three numeric preprocessing phases
    (imputation, outlier handling, scaling) by inspecting the behavioral flags of
    the selected strategies.

    The three preprocessing phases are not order-independent. Depending on which
    strategies are selected, applying them in the wrong order can silently corrupt
    statistics or produce meaningless results:

      - Outlier detection (IQR, Z-score, Winsorize) operates on raw values and
        cannot handle NaN — they must either receive imputed data, or the conflict
        must be resolved by running outlier handling first when imputation itself
        would be distorted by outliers.

      - Mean imputation and boundary imputation compute mean/std from the data.
        If extreme outliers are still present, these statistics are biased and the
        imputed values will be wrong. These strategies need clean data first.

      - Scalers like StandardScaler and MinMaxScaler are highly sensitive to outliers.
        A single extreme value can collapse the entire scaled range. These must run
        after outlier handling.

    Rather than hardcoding a fixed order or adding it as a separate Optuna parameter
    (which would explode the search space with invalid combinations), the order is
    derived deterministically from the strategy flags. This keeps the pipeline
    self-consistent: the strategies themselves declare their requirements, and the
    order follows automatically.

    Examples:
        MedianImputation + IQRStrategy + RobustScalerStrategy
        → [IMPUTATION, OUTLIER, SCALING]
        Median is outlier-safe, RobustScaler is outlier-safe → canonical order.

        MeanImputation + IQRStrategy + StandardScalerStrategy
        → [OUTLIER, IMPUTATION, SCALING]
        Mean stats are distorted by outliers → outlier wins the conflict.
        StandardScaler is outlier-sensitive → outlier before scaling.

        MedianImputation + ZScoreStrategy + StandardScalerStrategy
        → [IMPUTATION, OUTLIER, SCALING]
        Median is safe → canonical imputation first.
        StandardScaler still needs clean data → outlier before scaling (already satisfied).
    """
    nodes = {_PHASE.IMPUTATION, _PHASE.OUTLIER, _PHASE.SCALING}
    edges: Set[Tuple[_PHASE, _PHASE]] = set()

    # unconditional
    edges.add((_PHASE.IMPUTATION, _PHASE.SCALING))

    # potential conflict
    needs_clean = imputation.requires_outliers_removed  # OUTLIER -> IMPUTATION

    if needs_clean:
        edges.add((_PHASE.OUTLIER, _PHASE.IMPUTATION))
    else:
        edges.add((_PHASE.IMPUTATION, _PHASE.OUTLIER))

    # sensitive scalers must receive outlier-free data
    if scaling.requires_outliers_removed:
        edges.add((_PHASE.OUTLIER, _PHASE.SCALING))

    return _topological_sort(nodes, edges)
