from typing import ClassVar, cast

import numpy as np
import pandas as pd
from scipy import stats as sstats

from pills_core._enums import SemanticRole, TransformPhase
from pills_core.strategies.base import ColumnMeta
from pills_core.strategies.numeric.base import NumericalStrategy
from pills_core.strategies.score import DecisionScore
from pills_core.types.stats import NumericalColumnStats


class NumericalScalingStrategy(NumericalStrategy):
    requires_non_negative: ClassVar[bool] = False  # LogTransform, SqrtTransform
    is_invertible: ClassVar[bool] = True  # whether inverse denormalization is possible
    preserves_distribution: ClassVar[bool] = (
        True  # StandardScaler/Robust; False for Log
    )
    sensitive_to_outliers: ClassVar[bool] = False
    assumes_normality: ClassVar[bool] = False

    @property
    def phase(self) -> TransformPhase:
        return TransformPhase.SCALING

    def should_apply(self, stats: NumericalColumnStats, meta: ColumnMeta) -> bool:
        if meta.is_target:
            return False
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
        if self.sensitive_to_outliers and stats.outlier_ratio >= 0.05:
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
    sensitive_to_outliers = True
    assumes_normality = True
    preserves_distribution = True
    is_invertible = True

    def __init__(self) -> None:
        super().__init__(name="standard_scaler")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return (data - stats.mean) / stats.std

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = _semantic_role_penalty(meta)

        if abs(stats.skewness) < 0.5:
            condition += 100  # best when symmetric — mean is accurate center

        if abs(stats.skewness) >= 1.0:
            penalty += 100  # skew distorts mean → center is wrong

        if abs(stats.skewness) < 0.5 and stats.outlier_ratio < 0.08:
            condition += 75

        if stats.outlier_ratio < 0.02:
            condition += 75  # clean data — mean/std are reliable

        if stats.outlier_ratio >= 0.05:
            penalty += 150  # outliers destroy mean and std completely

        if meta.profile.is_heavy_tailed:
            penalty += 75  # heavy tails inflate std → unit variance is misleading

        if not meta.profile.is_skewed and not meta.profile.is_heavy_tailed:
            condition += 50  # well-behaved bell curve — standard scaler is ideal

        if meta.profile.is_low_variance:
            penalty += 40  # near-zero std → division becomes numerically unstable

        if meta.semantic_role == SemanticRole.CONTINUOUS:
            condition += (
                50  # standard scaler is the canonical choice for continuous features
            )

        if meta.semantic_role == SemanticRole.COUNT:
            penalty += (
                50  # counts are skewed and non-negative — mean/std are unreliable
            )

        return DecisionScore(base=250, condition=condition, penalty=penalty)


class MinMaxScalerStrategy(NumericalScalingStrategy):
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

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = _semantic_role_penalty(meta)

        if stats.outlier_ratio == 0:
            condition += 125  # min/max are accurate only without outliers

        if stats.outlier_ratio > 0:
            penalty += 150  # even one outlier collapses the [0,1] range

        if abs(stats.skewness) < 0.5:
            condition += 50

        if meta.profile.is_heavy_tailed:
            penalty += (
                100  # extreme values dominate the range — most values compress near 0
            )

        if meta.profile.is_sparse:
            penalty += (
                50  # range computed over few values — min/max are not representative
            )

        if meta.profile.is_low_variance:
            condition += (
                40  # tight cluster with no outliers — min/max scaling works well
            )

        if meta.semantic_role == SemanticRole.CONTINUOUS:
            condition += 30

        if meta.semantic_role == SemanticRole.COUNT:
            penalty += 40  # counts often have extreme max values — range gets dominated

        return DecisionScore(base=200, condition=condition, penalty=penalty)


class LogTransformStrategy(NumericalScalingStrategy):
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

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = _semantic_role_penalty(meta)

        if stats.skewness > 3.0:
            condition += 150  # heavy right tail — log is ideal

        elif stats.skewness > 1.5:
            condition += 100  # moderate right tail — log helps

        if stats.min == 0:
            condition += 25  # log1p(0) = 0, zero-safe

        if stats.outlier_ratio > 0.1:
            penalty += 50  # log compresses but doesn't remove outliers

        if meta.profile.is_heavy_tailed:
            condition += 75  # log is specifically designed to tame heavy right tails

        if meta.profile.is_sparse:
            penalty += 30  # few points — hard to verify skew is structural, not noise

        if meta.semantic_role == SemanticRole.COUNT:
            condition += (
                175  # counts follow Poisson/power-law — log transform is canonical
            )

        if meta.semantic_role == SemanticRole.COUNT and stats.zero_ratio > 0.5:
            condition += 50  # zero-inflated count — log1p

        if meta.semantic_role == SemanticRole.CONTINUOUS:
            condition += 30  # valid for right-skewed continuous features

        if meta.semantic_role == SemanticRole.ORDINAL:
            penalty += 50  # log distorts rank spacing in a non-meaningful way

        return DecisionScore(base=150, condition=condition, penalty=penalty)


class RobustScalerStrategy(NumericalScalingStrategy):
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

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = _semantic_role_penalty(meta)

        if stats.outlier_ratio > 0.05:
            condition += 100  # median/IQR not affected by outliers

        if meta.profile.is_heavy_tailed:
            condition += 75  # robust scaler shines when tails are extreme

        if meta.profile.has_outliers:
            condition += (
                50  # confirmed outliers — median/IQR centering is the right call
            )

        if meta.profile.is_low_variance:
            penalty += 60  # IQR near zero → division becomes unstable, same as std=0

        if meta.semantic_role == SemanticRole.CONTINUOUS:
            condition += 40  # safe general-purpose choice when outliers are present

        if meta.semantic_role == SemanticRole.COUNT:
            condition += 20  # right-skewed counts with extreme values — robust scaler handles well

        if abs(stats.skewness) > 2.0 and stats.outlier_ratio > 0.05:
            penalty += 300

        return DecisionScore(base=220, condition=condition, penalty=penalty)


class BoxCoxStrategy(NumericalScalingStrategy):
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

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = _semantic_role_penalty(meta)

        if stats.skewness > 4.0:
            condition += 175

        elif stats.skewness > 2.0:
            condition += 100

        if meta.profile.is_heavy_tailed:
            condition += 100

        if stats.outlier_ratio > 0.1:
            penalty += 75

        if meta.profile.has_outliers:
            penalty += 50

        if meta.semantic_role == SemanticRole.CONTINUOUS:
            condition += 50

        if meta.semantic_role == SemanticRole.COUNT:
            penalty += 125

        if meta.semantic_role == SemanticRole.ORDINAL:
            penalty += 100

        return DecisionScore(base=220, condition=condition, penalty=penalty)


class SqrtTransformStrategy(NumericalScalingStrategy):
    requires_non_negative = True
    is_invertible = True
    preserves_distribution = False
    sensitive_to_outliers = True

    def __init__(self) -> None:
        super().__init__(name="sqrt_transform")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return pd.Series(np.sqrt(data), index=data.index)

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = _semantic_role_penalty(meta)

        if 1.0 < stats.skewness <= 2.5:
            condition += 150

        elif stats.skewness > 2.5:
            condition += 50

        if meta.profile.has_outliers:
            penalty += 50

        if meta.semantic_role == SemanticRole.COUNT:
            condition += 80

        if meta.semantic_role == SemanticRole.ORDINAL:
            penalty += 100

        return DecisionScore(base=200, condition=condition, penalty=penalty)


def _semantic_role_penalty(meta: ColumnMeta) -> int:
    """
    some semantic roles make scaling actively harmful
    regardless of the chosen strategy.
    Returns a penalty to subtract from the score.
    """
    role = meta.semantic_role
    penalty = 0

    # IDs have no meaningful numeric distribution — scaling is nonsensical
    if role == SemanticRole.ID_LIKE:
        penalty += 300

    # Binary columns are already in {0, 1} — scaling distorts the encoding
    if role == SemanticRole.BINARY:
        penalty += 300

    # Nominal numbers encode categories — scaling implies false ordering
    if role == SemanticRole.NUMERIC_NOMINAL:
        penalty += 300

    # Ordinal ranks have meaningful order but not meaningful magnitude
    if role == SemanticRole.ORDINAL:
        penalty += 75

    return penalty
