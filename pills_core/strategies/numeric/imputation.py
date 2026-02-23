from pills_core._enums import TransformPhase
from pills_core.strategies.numeric.base import NumericalStrategy
from pills_core.strategies.priorities import for_lower_boundary, for_mean, for_median, for_mode, for_upper_boundary, for_zero
from pills_core.types.stats import NumericalColumnStats
import pandas as pd


class NumericalImputationStrategy(NumericalStrategy):
    fills_with_existing_value: bool = True # it uses central tendency or a constant
    sensitive_to_outliers: bool = False # performance degrades with outliers
    sensitive_to_skewness: bool = False # it's unsuitable for skewed distributions
    preserves_distribution: bool = True # the data shape remains relatively intact
    safe_for_target: bool = True # it's safe to use on the target variable (y)

    @property
    def phase(self) -> TransformPhase:
        return TransformPhase.IMPUTATION

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        if not self.safe_for_target and is_target:
            return False
        if self.sensitive_to_outliers and stats.outlier_ratio > 0.05:
            return False
        if self.sensitive_to_skewness and abs(stats.skewness) > 1.0:
            return False
        return stats.missing_ratio > 0

    def explain(self, stats: NumericalColumnStats) -> str:
        parts = [f"Imputing {stats.missing_ratio:.1%} missing with '{self.name}'"]
        if not self.preserves_distribution:
            parts.append("distorts distribution")
        if self.sensitive_to_outliers:
            parts.append("sensitive to outliers")
        if self.sensitive_to_skewness:
            parts.append("sensitive to skewness")
        if not self.safe_for_target:
            parts.append("unsafe for target")
        return " | ".join(parts)


class MedianImputation(NumericalImputationStrategy):
    fills_with_existing_value = True
    sensitive_to_outliers = False
    sensitive_to_skewness = False
    preserves_distribution = True

    def __init__(self) -> None:
        super().__init__(name="median")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.median)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_median(stats))


class MeanImputation(NumericalImputationStrategy):
    fills_with_existing_value = True
    sensitive_to_outliers = True
    sensitive_to_skewness = True
    preserves_distribution = True

    def __init__(self) -> None:
        super().__init__(name="mean")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mean)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_mean(stats))


class ModeImputation(NumericalImputationStrategy):
    fills_with_existing_value = True
    sensitive_to_outliers = False
    sensitive_to_skewness = False
    preserves_distribution = False  # artificially inflates the mode frequency

    def __init__(self) -> None:
        super().__init__(name="mode")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mode)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_mode(stats))


class ZeroImputation(NumericalImputationStrategy):
    fills_with_existing_value = False
    sensitive_to_outliers = False
    sensitive_to_skewness = False
    preserves_distribution = False

    def __init__(self) -> None:
        super().__init__(name="constant_zero")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(0)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_zero(stats))


class UpperBoundaryImputation(NumericalImputationStrategy):
    fills_with_existing_value = True
    sensitive_to_outliers = True
    sensitive_to_skewness = False
    preserves_distribution = False
    safe_for_target = False

    def __init__(self) -> None:
        super().__init__(name="upper_boundary")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mean + 3 * stats.std)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_upper_boundary(stats))


class LowerBoundaryImputation(NumericalImputationStrategy):
    fills_with_existing_value = True
    sensitive_to_outliers = True
    sensitive_to_skewness = False
    preserves_distribution = False
    safe_for_target = False

    def __init__(self) -> None:
        super().__init__(name="lower_boundary")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mean - 3 * stats.std)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_lower_boundary(stats))