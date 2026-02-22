from typing import Callable

import pandas as pd

from pills_core.calculations.fill_fn import (
    lower_boundary_fill,
    mean_fill,
    median_fill,
    mode_fill,
    upper_boundary_fill,
    zero_fill,
)
from pills_core.calculations.score_fn import (
    lower_boundary_score,
    mean_score,
    median_score,
    mode_score,
    upper_boundary_score,
    zero_score,
)
from pills_core.strategies.numeric.base import NumericalStrategy
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
    fill_fn=median_fill,
    score_fn=median_score,
)

MEAN_IMPUTATION = ImputationStrategy(
    name="mean",
    fill_fn=mean_fill,
    score_fn=mean_score,
)

MODE_IMPUTATION = ImputationStrategy(
    name="mode",
    fill_fn=mode_fill,
    score_fn=mode_score,
)

ZERO_IMPUTATION = ImputationStrategy(
    name="constant_zero",
    fill_fn=zero_fill,
    score_fn=zero_score,
)

UPPER_BOUNDARY_IMPUTATION = ImputationStrategy(
    name="upper_boundary",
    fill_fn=upper_boundary_fill,
    score_fn=upper_boundary_score,
)

LOWER_BOUNDARY_IMPUTATION = ImputationStrategy(
    name="lower_boundary",
    fill_fn=lower_boundary_fill,
    score_fn=lower_boundary_score,
)
