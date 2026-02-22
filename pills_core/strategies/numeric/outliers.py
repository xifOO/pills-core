import pandas as pd

from pills_core.strategies.numeric.base import NumericalStrategy
from pills_core.strategies.priorities import Priority
from pills_core.types.stats import NumericalColumnStats


class IQRStrategy(NumericalStrategy):
    def __init__(self) -> None:
        super().__init__(name="iqr")

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        return not is_target and abs(stats.skewness) < 1.5

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        iqr = stats.q3 - stats.q1
        return data.clip(lower=stats.q1 - 1.5 * iqr, upper=stats.q3 + 1.5 * iqr)

    def priority(self, stats: NumericalColumnStats) -> int:
        symmetric = abs(stats.skewness) < 1.0
        few_outliers = stats.outlier_ratio < 0.05

        if symmetric and few_outliers:
            return Priority.OUTLIER_HIGH
        elif symmetric or few_outliers:
            return Priority.OUTLIER_MID
        else:
            return Priority.OUTLIER_LOW


class WinsorizeStrategy(NumericalStrategy):
    def __init__(self, lower: float = 0.05, upper: float = 0.95) -> None:
        super().__init__(name="winsorize")
        self.lower = lower
        self.upper = upper

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        return not is_target and stats.outlier_ratio > 0.01

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        lo = data.quantile(self.lower)
        hi = data.quantile(self.upper)
        return data.clip(lo, hi)

    def priority(self, stats: NumericalColumnStats) -> int:
        skewed = abs(stats.skewness) >= 1.0
        many_outliers = stats.outlier_ratio >= 0.05

        if skewed and many_outliers:
            return Priority.OUTLIER_HIGH
        elif skewed or many_outliers:
            return Priority.OUTLIER_MID
        else:
            return Priority.OUTLIER_LOW
