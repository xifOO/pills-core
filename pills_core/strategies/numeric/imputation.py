from typing import ClassVar

from pills_core._enums import FamilyRole, TransformPhase
from pills_core.strategies.base import ColumnMeta, StrategyEmbedding
from pills_core.strategies.numeric.base import NumericalStrategy
from pills_core.types.stats import NumericalColumnStats
import pandas as pd


class NumericalImputationStrategy(NumericalStrategy):
    fills_with_existing_value: ClassVar[bool] = (
        True  # it uses central tendency or a constant
    )
    sensitive_to_outliers: ClassVar[bool] = False  # performance degrades with outliers
    sensitive_to_skewness: ClassVar[bool] = (
        False  # it's unsuitable for skewed distributions
    )
    preserves_distribution: ClassVar[bool] = (
        True  # the data shape remains relatively intact
    )
    safe_for_target: ClassVar[bool] = (
        True  # it's safe to use on the target variable (y)
    )

    @property
    def phase(self) -> TransformPhase:
        return TransformPhase.IMPUTATION

    def should_apply(self, stats: NumericalColumnStats, meta: ColumnMeta) -> bool:
        if not self.safe_for_target and meta.is_target:
            return False
        if self.sensitive_to_outliers and stats.outlier_ratio > 0.15:
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
    family_role: ClassVar[FamilyRole] = FamilyRole.CENTRAL_TENDENCY

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.9,
        outliers_sensitivity=0.1,
        missing_ratio_fit=0.8,
        distribution_preservation=0.9,
        target_safety=1.0,
        cardinality_fit=0.0,
    )
    radius = 1.9

    fills_with_existing_value = True
    sensitive_to_outliers = False
    sensitive_to_skewness = False
    preserves_distribution = True

    def __init__(self) -> None:
        super().__init__(name="median")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.median)


class MeanImputation(NumericalImputationStrategy):
    family_role: ClassVar[FamilyRole] = FamilyRole.CENTRAL_TENDENCY

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.2,
        outliers_sensitivity=0.9,
        missing_ratio_fit=0.7,
        distribution_preservation=0.8,
        target_safety=1.0,
        cardinality_fit=0.5,
    )
    radius = 1.2

    fills_with_existing_value = True
    sensitive_to_outliers = True
    sensitive_to_skewness = True
    preserves_distribution = True

    def __init__(self) -> None:
        super().__init__(name="mean")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mean)


class ModeImputation(NumericalImputationStrategy):
    family_role: ClassVar[FamilyRole] = FamilyRole.CENTRAL_TENDENCY

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.5,
        outliers_sensitivity=0.1,
        missing_ratio_fit=0.6,
        distribution_preservation=0.3,
        target_safety=1.0,
        cardinality_fit=0.9,
    )
    radius = 1.2

    fills_with_existing_value = True
    sensitive_to_outliers = False
    sensitive_to_skewness = False
    preserves_distribution = False  # artificially inflates the mode frequency

    def __init__(self) -> None:
        super().__init__(name="mode")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mode)


class ZeroImputation(NumericalImputationStrategy):
    family_role: ClassVar[FamilyRole] = FamilyRole.CONSTANT

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.3,
        outliers_sensitivity=0.1,
        missing_ratio_fit=0.9,
        distribution_preservation=0.2,
        target_safety=1.0,
        cardinality_fit=0.4,
    )
    radius = 1.1

    fills_with_existing_value = False
    sensitive_to_outliers = False
    sensitive_to_skewness = False
    preserves_distribution = False

    def __init__(self) -> None:
        super().__init__(name="constant_zero")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(0)


class UpperBoundaryImputation(NumericalImputationStrategy):
    family_role: ClassVar[FamilyRole] = FamilyRole.BOUNDARY

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.8,
        outliers_sensitivity=0.6,
        missing_ratio_fit=0.5,
        distribution_preservation=0.2,
        target_safety=0.0,
        cardinality_fit=0.3,
    )
    radius = 1.1

    fills_with_existing_value = True
    sensitive_to_outliers = True
    sensitive_to_skewness = False
    preserves_distribution = False
    safe_for_target = False

    def __init__(self) -> None:
        super().__init__(name="upper_boundary")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mean + 3 * stats.std)


class LowerBoundaryImputation(NumericalImputationStrategy):
    family_role: ClassVar[FamilyRole] = FamilyRole.BOUNDARY

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.8,
        outliers_sensitivity=0.6,
        missing_ratio_fit=0.5,
        distribution_preservation=0.2,
        target_safety=0.0,
        cardinality_fit=0.3,
    )
    radius = 1.1

    fills_with_existing_value = True
    sensitive_to_outliers = True
    sensitive_to_skewness = False
    preserves_distribution = False
    safe_for_target = False

    def __init__(self) -> None:
        super().__init__(name="lower_boundary")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mean - 3 * stats.std)
