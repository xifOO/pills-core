from typing import Generic, TypeVar

import pandas as pd

from pills_core._enums import ColumnRole, TransformPhase
from pills_core.strategies.base import ColumnMeta, SingleStrategy

StrategyT = TypeVar("StrategyT", bound="SingleStrategy")


class StrategyRegistry(Generic[StrategyT]):
    MIN_SCORE_THRESHOLD = 200

    def __init__(self, column_type: ColumnRole, phase: TransformPhase) -> None:
        self.column_type = column_type
        self.phase = phase
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

    def resolve(self, meta: ColumnMeta, stats) -> list[StrategyT]:
        applicable = [
            s
            for s in self._strategies
            if s.should_apply(stats, meta)
            and s.score(meta, stats).total >= self.MIN_SCORE_THRESHOLD
        ]
        return sorted(
            applicable, key=lambda s: s.score(meta, stats).total, reverse=True
        )

    def apply(self, data: pd.Series, meta: ColumnMeta, stats) -> pd.Series:
        ordered = self.resolve(meta, stats)
        if not ordered:
            return data
        return ordered[0].apply(data, stats)

    def apply_all(self, data: pd.Series, meta: ColumnMeta, stats) -> pd.Series:
        result = data
        for strategy in self.resolve(meta, stats):
            result = strategy.apply(result, stats)
        return result

    def explain(self, meta: ColumnMeta, stats) -> list[str]:
        ordered = self.resolve(meta, stats)
        all_candidates = [s for s in self._strategies if s.should_apply(stats, meta)]

        lines = []

        # все кандидаты с их scores
        if all_candidates:
            scores_info = " | ".join(
                f"{s.name}={s.score(meta, stats).total}"
                for s in sorted(
                    all_candidates,
                    key=lambda s: s.score(meta, stats).total,
                    reverse=True,
                )
            )
            lines.append(f"  candidates: {scores_info}")

        if not ordered:
            return lines

        best = ordered[0]
        lines.append(
            f"[{self.phase.name}] {best.name} (score={best.score(meta, stats).total}): "
            f"{best.explain(stats)}"
        )
        return lines
