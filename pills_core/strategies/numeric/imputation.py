from typing import ClassVar

from pills_core._enums import SemanticRole, TransformPhase
from pills_core.strategies.base import ColumnMeta
from pills_core.strategies.numeric.base import NumericalStrategy
from pills_core.strategies.score import DecisionScore
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

    def _base_meta_penalty(self, meta: ColumnMeta) -> int:
        penalty = 0

        if not self.safe_for_target and meta.is_target:
            penalty += 500  # never for target

        if meta.semantic_role == SemanticRole.ID_LIKE:
            penalty += 500  # never for identifiers

        return penalty


class MedianImputation(NumericalImputationStrategy):
    fills_with_existing_value = True
    sensitive_to_outliers = False
    sensitive_to_skewness = False
    preserves_distribution = True

    def __init__(self) -> None:
        super().__init__(name="median")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.median)

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = self._base_meta_penalty(meta)

        if stats.missing_ratio == 0:
            penalty += 300

        if abs(stats.skewness) >= 1.0:
            condition += 100

        if stats.outlier_ratio > 0.05:
            condition += 75

        return DecisionScore(base=300, condition=condition, penalty=penalty)


class MeanImputation(NumericalImputationStrategy):
    fills_with_existing_value = True
    sensitive_to_outliers = True
    sensitive_to_skewness = True
    preserves_distribution = True

    def __init__(self) -> None:
        super().__init__(name="mean")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mean)

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = self._base_meta_penalty(meta)

        if abs(stats.skewness) < 0.5:
            condition += 100  # mean == median when symmetric

        if stats.outlier_ratio > 0.05:
            penalty += 150  # mean is pulled hard by outliers

        if abs(stats.skewness) >= 1.0:
            penalty += 100

        return DecisionScore(base=300, condition=condition, penalty=penalty)


class ModeImputation(NumericalImputationStrategy):
    fills_with_existing_value = True
    sensitive_to_outliers = False
    sensitive_to_skewness = False
    preserves_distribution = False  # artificially inflates the mode frequency

    def __init__(self) -> None:
        super().__init__(name="mode")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mode)

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = self._base_meta_penalty(meta)

        if stats.n_unique <= 10:
            condition += 100  # mode makes sense for low-cardinality

        if stats.n_unique > 20:
            penalty += 150

        return DecisionScore(base=300, condition=condition, penalty=penalty)


class ZeroImputation(NumericalImputationStrategy):
    fills_with_existing_value = False
    sensitive_to_outliers = False
    sensitive_to_skewness = False
    preserves_distribution = False

    def __init__(self) -> None:
        super().__init__(name="constant_zero")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(0)

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = self._base_meta_penalty(meta)

        if stats.missing_ratio > 0.3:
            condition += 75  # large gaps — zero is safe fallback

        if stats.min >= 0:
            condition += 25  # zero is in-domain for non-negative

        if stats.mean > 0 and stats.missing_ratio < 0.1:
            penalty += 100  # distorts distribution when data is mostly present

        return DecisionScore(base=150, condition=condition, penalty=penalty)


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

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = self._base_meta_penalty(meta)

        if stats.skewness > 2.0:
            condition += 100  # right tail — filling with upper boundary makes sense

        if stats.outlier_ratio > 0.05:
            condition += 75

        if stats.outlier_ratio > 0.15:
            penalty += 75  # boundary itself is distorted by too many outliers

        return DecisionScore(base=150, condition=condition, penalty=penalty)


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

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = self._base_meta_penalty(meta)

        if stats.skewness < -2.0:
            condition += 100  # left tail — filling with lower boundary makes sense

        if stats.outlier_ratio > 0.05:
            condition += 75

        if stats.outlier_ratio > 0.15:
            penalty += 75

        return DecisionScore(base=150, condition=condition, penalty=penalty)
