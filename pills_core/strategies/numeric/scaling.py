from typing import ClassVar, cast

import numpy as np
import pandas as pd
from scipy import stats as sstats

from pills_core._enums import FamilyRole, SemanticRole, TaskType, TransformPhase
from pills_core.strategies.numeric.base import (
    NumericalColumnMeta,
    NumericalEmbedding,
    NumericalStrategy,
)
from pills_core.types.stats import NumericalColumnStats


class NumericalScalingStrategy(NumericalStrategy):
    requires_non_negative: ClassVar[bool] = False  # LogTransform, SqrtTransform
    is_invertible: ClassVar[bool] = True  # whether inverse denormalization is possible
    preserves_distribution: ClassVar[bool] = (
        True  # StandardScaler/Robust; False for Log
    )
    sensitive_to_outliers: ClassVar[bool] = False
    assumes_normality: ClassVar[bool] = False
    requires_outliers_removed: ClassVar[bool] = False

    def __init__(
        self,
        *,
        embedding: NumericalEmbedding,
        radius: float,
        normality_skewness_limit: float = 1.0,
    ) -> None:
        super().__init__(embedding=embedding, radius=radius)
        self.normality_skewness_limit = normality_skewness_limit

    @property
    def phase(self) -> TransformPhase:
        return TransformPhase.SCALING

    def should_apply(
        self, stats: NumericalColumnStats, meta: NumericalColumnMeta
    ) -> bool:
        if meta.semantic_role in (
            SemanticRole.ID_LIKE,
            SemanticRole.BINARY,
            SemanticRole.NUMERIC_NOMINAL,
        ):
            return False
        if self.requires_non_negative and stats.min < 0:
            return False
        if (
            self.assumes_normality
            and abs(stats.skewness) >= self.normality_skewness_limit
        ):
            return False
        return True

    def ordering_constraints(
        self, present_phases: set[TransformPhase]
    ) -> set[tuple[TransformPhase, TransformPhase]]:
        if self.requires_outliers_removed and TransformPhase.OUTLIER in present_phases:
            return {(TransformPhase.OUTLIER, TransformPhase.SCALING)}
        return set()

    def explain(self, stats: NumericalColumnStats) -> str:
        parts = [f"Scaling with '{self.name}'"]
        parts.append(f"skewness={stats.skewness:.2f}")
        parts.append(f"outlier_ratio={stats.outlier_ratio:.1%}")
        if not self.preserves_distribution:
            parts.append("changes distribution shape")
        if not self.is_invertible:
            parts.append("not invertible")
        if self.sensitive_to_outliers:
            parts.append("sensitive to outliers")
        return " | ".join(parts)


class StandardScalerStrategy(NumericalScalingStrategy):
    name: ClassVar[str] = "standard_scaler"
    family_role: ClassVar[FamilyRole] = FamilyRole.LINEAR_SCALING

    sensitive_to_outliers: ClassVar[bool] = True
    assumes_normality: ClassVar[bool] = True
    preserves_distribution: ClassVar[bool] = True
    is_invertible: ClassVar[bool] = True
    requires_outliers_removed: ClassVar[bool] = True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return (data - stats.mean) / stats.std


class MinMaxScalerStrategy(NumericalScalingStrategy):
    name: ClassVar[str] = "min_max_scaler"
    family_role: ClassVar[FamilyRole] = FamilyRole.LINEAR_SCALING

    sensitive_to_outliers: ClassVar[bool] = (
        True  # → should_apply will block if outlier_ratio >= 0.05
    )
    assumes_normality: ClassVar[bool] = False
    preserves_distribution: ClassVar[bool] = True
    is_invertible: ClassVar[bool] = True

    def should_apply(
        self, stats: NumericalColumnStats, meta: NumericalColumnMeta
    ) -> bool:
        return super().should_apply(stats, meta) and (stats.max - stats.min) > 0

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        if meta.domain_profile.is_monetary:
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return (data - stats.min) / (stats.max - stats.min)


class LogTransformStrategy(NumericalScalingStrategy):
    name: ClassVar[str] = "log_transform"
    family_role: ClassVar[FamilyRole] = FamilyRole.SKEW_TRANSFORM

    requires_non_negative: ClassVar[bool] = True  # → should_apply will block if min < 0
    preserves_distribution: ClassVar[bool] = False
    sensitive_to_outliers: ClassVar[bool] = False
    is_invertible: ClassVar[bool] = True

    def __init__(
        self,
        *,
        embedding: NumericalEmbedding,
        radius: float,
        min_skewness: float,
    ) -> None:
        super().__init__(embedding=embedding, radius=radius)
        self.min_skewness = min_skewness

    def should_apply(
        self, stats: NumericalColumnStats, meta: NumericalColumnMeta
    ) -> bool:
        if meta.semantic_role == SemanticRole.COUNT and stats.min >= 0:
            return True
        return super().should_apply(stats, meta) and stats.skewness > self.min_skewness

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        if (
            meta.domain_profile.lower_bound is not None
            and meta.domain_profile.lower_bound < 0
        ):
            return False

        if meta.domain_profile.is_rate or meta.domain_profile.is_ratio:
            return False

        return True

    def is_task_valid(self, meta: NumericalColumnMeta) -> bool:
        if meta.is_target and meta.task_type == TaskType.BINARY:
            return False
        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return pd.Series(np.log1p(data), index=data.index)


class RobustScalerStrategy(NumericalScalingStrategy):
    name: ClassVar[str] = "robust_scaler"
    family_role: ClassVar[FamilyRole] = FamilyRole.ROBUST

    sensitive_to_outliers: ClassVar[bool] = False
    assumes_normality: ClassVar[bool] = False
    preserves_distribution: ClassVar[bool] = True
    is_invertible: ClassVar[bool] = True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        iqr = stats.q3 - stats.q1
        if iqr == 0:
            return data
        return (data - stats.median) / iqr


class BoxCoxStrategy(NumericalScalingStrategy):
    name: ClassVar[str] = "box_cox"
    family_role: ClassVar[FamilyRole] = FamilyRole.SKEW_TRANSFORM

    requires_non_negative: ClassVar[bool] = True  # Box-Cox requires positive values
    preserves_distribution: ClassVar[bool] = (
        False  # changes the shape of the distribution
    )
    sensitive_to_outliers: ClassVar[bool] = True
    is_invertible: ClassVar[bool] = False  # currently not invertible

    def __init__(
        self,
        *,
        embedding: NumericalEmbedding,
        radius: float,
        min_skewness: float,
        shift_epsilon: float,
    ) -> None:
        super().__init__(embedding=embedding, radius=radius)
        self.min_skewness = min_skewness
        self.shift_epsilon = shift_epsilon
        self.shift_: float = 0.0

    def should_apply(
        self, stats: NumericalColumnStats, meta: NumericalColumnMeta
    ) -> bool:
        if stats.skewness <= self.min_skewness:
            return False

        return super().should_apply(stats, meta)

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        if (
            meta.domain_profile.lower_bound is not None
            and meta.domain_profile.lower_bound < 0
        ):
            return False

        if meta.domain_profile.is_rate or meta.domain_profile.is_ratio:
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        values: np.ndarray = data.to_numpy(dtype=np.float64, copy=True)

        min_val: float = float(values.min())

        if min_val <= 0:
            self.shift_ = abs(min_val) + self.shift_epsilon
            values = values + self.shift_

        result = sstats.boxcox(values)

        transformed, _ = cast(tuple[np.ndarray, float], result)

        return pd.Series(transformed, index=data.index)


class SqrtTransformStrategy(NumericalScalingStrategy):
    name: ClassVar[str] = "sqrt_transform"
    family_role: ClassVar[FamilyRole] = FamilyRole.SKEW_TRANSFORM

    requires_non_negative: ClassVar[bool] = True
    is_invertible: ClassVar[bool] = True
    preserves_distribution: ClassVar[bool] = False
    sensitive_to_outliers: ClassVar[bool] = True

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        if meta.domain_profile.is_rate or meta.domain_profile.is_ratio:
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return pd.Series(np.sqrt(data), index=data.index)
