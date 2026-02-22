import pandas as pd

from pills_core.strategies.numeric.base import NumericalStrategy
from pills_core.strategies.priorities import Priority
from pills_core.types.stats import NumericalColumnStats


class OutlierStrategy(NumericalStrategy):
    pass


class IQRStrategy(OutlierStrategy):
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


class ZScoreStrategy(OutlierStrategy):
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        super().__init__(name="z-score")

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        return stats.std > 0 and abs(stats.skewness) < 1.0 and stats.outlier_ratio > 0

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        if stats.std == 0:
            return data

        lower = stats.mean - self.threshold * stats.std
        upper = stats.mean + self.threshold * stats.std

        return data.clip(lower=lower, upper=upper)

    def priority(self, stats: NumericalColumnStats) -> int:
        if abs(stats.skewness) < 0.5:
            return Priority.OUTLIER_HIGH
        return Priority.OUTLIER_LOW


class WinsorizeStrategy(OutlierStrategy):
    def __init__(self) -> None:
        super().__init__(name="winsorize")

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        return not is_target and stats.outlier_ratio > 0.01

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.clip(stats.p05, stats.p95)

    def priority(self, stats: NumericalColumnStats) -> int:
        skewed = abs(stats.skewness) >= 1.0
        many_outliers = stats.outlier_ratio >= 0.05

        if skewed and many_outliers:
            return Priority.OUTLIER_HIGH
        elif skewed or many_outliers:
            return Priority.OUTLIER_MID
        else:
            return Priority.OUTLIER_LOW
