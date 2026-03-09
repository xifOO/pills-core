from typing import Dict, Generic, List, TypeVar

import pandas as pd

from pills_core._enums import ColumnRole, TransformPhase
from pills_core.strategies.base import ColumnMeta, SingleStrategy, StrategyEmbedding

StrategyT = TypeVar("StrategyT", bound="SingleStrategy")


class StrategyRegistry(Generic[StrategyT]):
    def __init__(
        self,
        column_type: ColumnRole,
        phase: TransformPhase,
        weights: Dict[str, float],
    ) -> None:
        self.column_type = column_type
        self.phase = phase
        self.weights = weights
        self._strategies: list[StrategyT] = []

    def register(self, strategy: StrategyT) -> "StrategyRegistry[StrategyT]":
        if strategy.column_type != self.column_type:
            raise TypeError(
                f"Strategy {strategy.__class__.__name__} should be for {strategy.column_type}."
            )
        if strategy.phase != self.phase:
            raise TypeError(
                f"Strategy '{strategy.name}' belongs to phase {strategy.phase}."
            )
        self._strategies.append(strategy)
        return self

    def resolve(
        self, meta: ColumnMeta, column_embedding: StrategyEmbedding, stats
    ) -> list[tuple[StrategyT, float]]:
        candidates = []

        for s in self._strategies:
            if s.should_apply(stats, meta):
                dist = s.distance(column_embedding, self.weights)

                if dist <= s.radius:
                    candidates.append((s, dist))

        candidates.sort(key=lambda x: x[1])
        return candidates

    def apply(
        self,
        data: pd.Series,
        meta: ColumnMeta,
        column_embedding: StrategyEmbedding,
        stats,
    ) -> pd.Series:
        ordered = self.resolve(meta, column_embedding, stats)
        if not ordered:
            return data
        return ordered[0][0].apply(data, stats)

    def apply_all(
        self,
        data: pd.Series,
        meta: ColumnMeta,
        column_embedding: StrategyEmbedding,
        stats,
    ) -> pd.Series:
        result = data
        for strategy in self.resolve(meta, column_embedding, stats):
            result = strategy[0].apply(result, stats)
        return result

    def get_search_space(self, stats) -> Dict[str, List[str]]:
        """Remove out all impossible strategy combinations
        before Optuna even tries them."""
        ...

    def explain(
        self, meta: ColumnMeta, column_embedding: StrategyEmbedding, stats
    ) -> list[str]:
        ordered = self.resolve(meta, column_embedding, stats)
        lines = []

        all_distances = []
        for s in self._strategies:
            applies = s.should_apply(stats, meta)
            dist = s.distance(column_embedding, self.weights)
            in_radius = dist <= s.radius
            status = (
                "✓"
                if (applies and in_radius)
                else ("✗ radius" if applies else "✗ should_apply")
            )
            all_distances.append((s, dist, status))
        all_distances.sort(key=lambda x: x[1])

        lines.append("  all strategies:")
        for s, dist, status in all_distances:
            lines.append(f"    {status} {s.name}: d={dist:.3f} r={s.radius}")

        if ordered:
            candidates_info = " | ".join(
                f"{s.name}=d:{dist:.3f}(r:{s.radius})" for s, dist in ordered
            )
            lines.append(f"  candidates: {candidates_info}")
            best_s, best_dist = ordered[0]
            lines.append(
                f"[{self.phase.name}] {best_s.name} "
                f"(distance={best_dist:.3f}, radius={best_s.radius}): "
                f"{best_s.explain(stats)}"
            )
        else:
            lines.append(f"[{self.phase.name}] No applicable strategies found.")

        return lines
