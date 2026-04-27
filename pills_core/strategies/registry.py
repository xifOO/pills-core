from typing import Dict, List, Optional, Tuple

from pills_core._enums import ColumnRole, TransformPhase
from pills_core.strategies.base import ColumnMeta, SingleStrategy, StrategyEmbedding
from pills_core.types.stats import BaseColumnStats


class StrategyRegistry:
    def __init__(
        self,
        column_type: ColumnRole,
        phase: TransformPhase,
        weights: Dict[str, float],
    ) -> None:
        self.column_type = column_type
        self.phase = phase
        self.weights = weights
        self._strategies: List[SingleStrategy] = []

    @property
    def strategies(self) -> List[SingleStrategy]:
        return self._strategies

    def _register(self, strategy: SingleStrategy) -> "StrategyRegistry":
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

    def bulk_register(
        self, strategies: List[Optional[SingleStrategy]]
    ) -> "StrategyRegistry":
        for s in strategies:
            if s is not None:
                self._register(s)
        return self

    def resolve(
        self,
        meta: ColumnMeta,
        column_embedding: StrategyEmbedding,
        stats: BaseColumnStats,
    ) -> List[Tuple[SingleStrategy, float]]:

        scored: List[Tuple[SingleStrategy, float]] = []

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

    def get_search_space(
        self,
        meta: ColumnMeta,
        column_embedding: StrategyEmbedding,
        stats: BaseColumnStats,
    ) -> Dict[str, List[str]]:
        """Remove out all impossible strategy combinations before Optuna even tries them."""
        return {
            self.phase.value: [
                s.name for s, _ in self.resolve(meta, column_embedding, stats)
            ]
        }
