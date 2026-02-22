from typing import Generic, TypeVar

import pandas as pd

from pills_core._enums import ColumnRole
from pills_core.strategies.base import SingleStrategy


StrategyT = TypeVar("StrategyT", bound="SingleStrategy")


class StrategyRegistry(Generic[StrategyT]):
    def __init__(self, column_type: ColumnRole) -> None:
        self.column_type = column_type
        self._strategies: list[StrategyT] = []

    def register(self, strategy: StrategyT) -> "StrategyRegistry[StrategyT]":
        if strategy.column_type != self.column_type:
            raise TypeError(
                f"Strategy {strategy.__class__.__name__} should be for {strategy.column_type}."
            )
        self._strategies.append(strategy)
        return self

    def resolve(self, stats, is_target: bool) -> list[StrategyT]:
        applicable = [s for s in self._strategies if s.should_apply(stats, is_target)]
        return sorted(applicable, key=lambda s: s.priority(stats), reverse=True)

    def apply(self, data: pd.Series, stats, is_target: bool) -> pd.Series:
        ordered = self.resolve(stats, is_target)

        if not ordered:
            return data

        result = data
        for strategy in ordered:
            result = strategy.apply(result, stats)
        return result

    def explain(self, stats, is_target: bool) -> list[str]:
        ordered = self.resolve(stats, is_target)
        return [
            f"{i + 1}. {s.__class__.__name__} (priority={s.priority(stats)})"
            for i, s in enumerate(ordered)
        ]
