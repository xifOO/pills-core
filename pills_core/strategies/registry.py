from typing import Dict, Generic, List, Tuple, TypeVar

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
        self._strategies: List[StrategyT] = []

    @property
    def strategies(self) -> List[StrategyT]:
        # now only for tests
        return self._strategies

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
    ) -> List[Tuple[StrategyT, float]]:

        scored: List[Tuple[StrategyT, float]] = []

        for strategy in self._strategies:
            score = strategy.score(
                column_embedding=column_embedding,
                stats=stats,
                meta=meta,
                weights=self.weights,
            )

            if score is not None:
                scored.append((strategy, score))

        scored.sort(key=lambda x: x[1])
        return scored

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

    def get_search_space(
        self, meta: ColumnMeta, column_embedding: StrategyEmbedding, stats
    ) -> Dict[str, List[str]]:
        """Remove out all impossible strategy combinations before Optuna even tries them."""
        return {
            self.phase.value: [
                s.name for s, _ in self.resolve(meta, column_embedding, stats)
            ]
        }

    def explain(
        self, meta: ColumnMeta, column_embedding: StrategyEmbedding, stats
    ) -> List[str]:
        ordered = self.resolve(meta, column_embedding, stats)
        lines = []

        all_distances = []
        for s in self._strategies:
            applies = s.should_apply(stats, meta)
            domain_ok = s.is_domain_valid(meta)
            dist = s.distance(column_embedding, self.weights)
            in_radius = dist <= s.radius

            status = (
                "✓"
                if (applies and domain_ok and in_radius)
                else "✗ radius"
                if (applies and domain_ok)
                else "✗ domain"
                if (applies and not domain_ok)
                else "✗ should_apply"
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
