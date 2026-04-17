from typing import Dict, Generic, List, Tuple, TypeVar

from pills_core._enums import ColumnRole, TransformPhase
from pills_core.strategies.base import MetaT, SingleStrategy, StrategyEmbedding
from pills_core.types.stats import StatsT

StrategyT = TypeVar("StrategyT", bound="SingleStrategy")


class StrategyRegistry(Generic[StrategyT, StatsT, MetaT]):
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

    def _register(
        self, strategy: StrategyT
    ) -> "StrategyRegistry[StrategyT, StatsT, MetaT]":
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

    def bulk_register(self, strategies: List[StrategyT]):
        for s in strategies:
            if s is not None:
                self._register(s)
        return self

    def resolve(
        self, meta: MetaT, column_embedding: StrategyEmbedding, stats: StatsT
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

    def get_search_space(
        self, meta: MetaT, column_embedding: StrategyEmbedding, stats: StatsT
    ) -> Dict[str, List[str]]:
        """Remove out all impossible strategy combinations before Optuna even tries them."""
        return {
            self.phase.value: [
                s.name for s, _ in self.resolve(meta, column_embedding, stats)
            ]
        }
