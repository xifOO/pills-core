from typing import ClassVar, cast

import numpy as np
import pandas as pd
from scipy import stats as sstats

from pills_core._enums import FamilyRole, SemanticRole, TransformPhase
from pills_core.strategies.base import ColumnMeta, StrategyEmbedding
from pills_core.strategies.numeric.base import NumericalStrategy
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

    @property
    def phase(self) -> TransformPhase:
        return TransformPhase.SCALING

    def should_apply(self, stats: NumericalColumnStats, meta: ColumnMeta) -> bool:
        if meta.semantic_role in (
            SemanticRole.ID_LIKE,
            SemanticRole.BINARY,
            SemanticRole.NUMERIC_NOMINAL,
        ):
            return False
        if self.requires_non_negative and stats.min < 0:
            return False
        if self.assumes_normality and abs(stats.skewness) >= 1.0:
            return False
        return True

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
    family_role: ClassVar[FamilyRole] = FamilyRole.LINEAR_SCALING

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.1,
        outliers_sensitivity=0.9,
        missing_ratio_fit=0.5,
        distribution_preservation=1.0,
        target_safety=0.0,
        cardinality_fit=0.4,
    )
    radius = 1.2

    sensitive_to_outliers = True
    assumes_normality = True
    preserves_distribution = True
    is_invertible = True
    requires_outliers_removed = True

    def __init__(self) -> None:
        super().__init__(name="standard_scaler")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return (data - stats.mean) / stats.std


class MinMaxScalerStrategy(NumericalScalingStrategy):
    family_role: ClassVar[FamilyRole] = FamilyRole.LINEAR_SCALING

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.3,
        outliers_sensitivity=1.0,
        missing_ratio_fit=0.5,
        distribution_preservation=1.0,
        target_safety=0.0,
        cardinality_fit=0.4,
    )
    radius = 1.0

    sensitive_to_outliers = True  # → should_apply will block if outlier_ratio >= 0.05
    assumes_normality = False
    preserves_distribution = True
    is_invertible = True

    def __init__(self) -> None:
        super().__init__(name="min_max_scaler")

    def should_apply(self, stats: NumericalColumnStats, meta: ColumnMeta) -> bool:
        return super().should_apply(stats, meta) and (stats.max - stats.min) > 0

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return (data - stats.min) / (stats.max - stats.min)


class LogTransformStrategy(NumericalScalingStrategy):
    family_role: ClassVar[FamilyRole] = FamilyRole.SKEW_TRANSFORM

    embedding = StrategyEmbedding(
        skewness_sensitivity=1.0,
        outliers_sensitivity=0.3,
        missing_ratio_fit=0.5,
        distribution_preservation=0.1,
        target_safety=0.0,
        cardinality_fit=0.6,
    )
    radius = 1.1

    requires_non_negative = True  # → should_apply will block if min < 0
    preserves_distribution = False
    sensitive_to_outliers = False
    is_invertible = True

    def __init__(self):
        super().__init__(name="log_transform")

    def should_apply(self, stats: NumericalColumnStats, meta: ColumnMeta) -> bool:
        if meta.semantic_role == SemanticRole.COUNT and stats.min >= 0:
            return True
        return super().should_apply(stats, meta) and stats.skewness > 1.5

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return pd.Series(np.log1p(data), index=data.index)


class RobustScalerStrategy(NumericalScalingStrategy):
    family_role: ClassVar[FamilyRole] = FamilyRole.ROBUST

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.5,
        outliers_sensitivity=0.1,
        missing_ratio_fit=0.5,
        distribution_preservation=0.9,
        target_safety=0.0,
        cardinality_fit=0.4,
    )
    radius = 1.5

    sensitive_to_outliers = False
    assumes_normality = False
    preserves_distribution = True
    is_invertible = True

    def __init__(self) -> None:
        super().__init__(name="robust_scaler")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        iqr = stats.q3 - stats.q1
        if iqr == 0:
            return data
        return (data - stats.median) / iqr


class BoxCoxStrategy(NumericalScalingStrategy):
    family_role: ClassVar[FamilyRole] = FamilyRole.SKEW_TRANSFORM

    embedding = StrategyEmbedding(
        skewness_sensitivity=1.0,
        outliers_sensitivity=0.7,
        missing_ratio_fit=0.5,
        distribution_preservation=0.1,
        target_safety=0.0,
        cardinality_fit=0.3,
    )
    radius = 1.0

    requires_non_negative = True  # Box-Cox requires positive values
    preserves_distribution = False  # changes the shape of the distribution
    sensitive_to_outliers = True
    is_invertible = False  # currently not invertible

    def __init__(self) -> None:
        super().__init__(name="box_cox")
        self.shift_: float = 0.0

    def should_apply(self, stats: NumericalColumnStats, meta: ColumnMeta) -> bool:
        if stats.skewness <= 1.5:
            return False

        return super().should_apply(stats, meta)

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        values: np.ndarray = data.to_numpy(dtype=np.float64, copy=True)

        min_val: float = float(values.min())

        if min_val <= 0:
            self.shift_ = abs(min_val) + 1e-6
            values = values + self.shift_

        result = sstats.boxcox(values)

        transformed, _ = cast(tuple[np.ndarray, float], result)

        return pd.Series(transformed, index=data.index)


class SqrtTransformStrategy(NumericalScalingStrategy):
    family_role: ClassVar[FamilyRole] = FamilyRole.SKEW_TRANSFORM

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.7,
        outliers_sensitivity=0.5,
        missing_ratio_fit=0.5,
        distribution_preservation=0.3,
        target_safety=0.0,
        cardinality_fit=0.5,
    )
    radius = 1.2

    requires_non_negative = True
    is_invertible = True
    preserves_distribution = False
    sensitive_to_outliers = True

    def __init__(self) -> None:
        super().__init__(name="sqrt_transform")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return pd.Series(np.sqrt(data), index=data.index)
