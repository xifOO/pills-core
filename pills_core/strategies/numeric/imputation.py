from typing import Callable

import pandas as pd

from pills_core.strategies.numeric.base import NumericalStrategy
from pills_core.strategies.priorities import Priority
from pills_core.types.stats import NumericalColumnStats


class ImputationStrategy(NumericalStrategy):
    def __init__(
        self,
        name: str,
        fill_fn: Callable[[pd.Series, NumericalColumnStats], pd.Series],
        score_fn: Callable[[NumericalColumnStats], int],
    ) -> None:
        self.name = name
        self._fill_fn = fill_fn
        self._score_fn = score_fn

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        return stats.missing_ratio > 0

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return self._fill_fn(data, stats)

    def priority(self, stats: NumericalColumnStats) -> int:
        return self._score_fn(stats)


MEDIAN_IMPUTATION = ImputationStrategy(
    name="median",
    fill_fn=lambda data, stats: data.fillna(stats.median),
    score_fn=lambda stats: (
        Priority.IMPUTATION_HIGH
        if abs(stats.skewness) >= 1.0 or stats.outlier_ratio > 0.05
        else Priority.IMPUTATION_LOW
    ),
)

MEAN_IMPUTATION = ImputationStrategy(
    name="mean",
    fill_fn=lambda data, stats: data.fillna(stats.mean),
    score_fn=lambda stats: (
        Priority.IMPUTATION_HIGH
        if abs(stats.skewness) < 1.0 and stats.outlier_ratio < 0.05
        else Priority.IMPUTATION_LOW
    ),
)

MODE_IMPUTATION = ImputationStrategy(
    name="mode",
    fill_fn=lambda data, stats: data.fillna(stats.mode),
    score_fn=lambda stats: (
        Priority.IMPUTATION_HIGH if stats.n_unique <= 10 else Priority.IMPUTATION_LOW
    ),
)


ZERO_IMPUTATION = ImputationStrategy(
    name="constant_zero",
    fill_fn=lambda data, stats: data.fillna(0),
    score_fn=lambda stats: (
        Priority.IMPUTATION_HIGH
        if stats.missing_ratio > 0.3 and stats.min >= 0
        else Priority.IMPUTATION_LOW
    ),
)


UPPER_BOUNDARY_IMPUTATION = ImputationStrategy(
    name="upper_boundary",
    fill_fn=lambda data, stats: data.fillna(stats.mean + 3 * stats.std),
    score_fn=lambda stats: (
        Priority.IMPUTATION_HIGH
        if stats.skewness > 2.0 and stats.outlier_ratio > 0.05
        else Priority.IMPUTATION_LOW
    ),
)


LOWER_BOUNDARY_IMPUTATION = ImputationStrategy(
    name="lower_boundary",
    fill_fn=lambda data, stats: data.fillna(stats.mean - 3 * stats.std),
    score_fn=lambda stats: (
        Priority.IMPUTATION_HIGH
        if stats.skewness < -2.0 and stats.outlier_ratio > 0.05
        else Priority.IMPUTATION_LOW
    ),
)
