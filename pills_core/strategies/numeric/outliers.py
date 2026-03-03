from typing import ClassVar

import pandas as pd

from pills_core._enums import SemanticRole, TransformPhase
from pills_core.strategies.base import ColumnMeta
from pills_core.strategies.numeric.base import NumericalStrategy
from pills_core.strategies.score import DecisionScore
from pills_core.types.stats import NumericalColumnStats


class NumericalOutlierStrategy(NumericalStrategy):
    clips_values: ClassVar[bool] = True  # strategy clips or removes values
    uses_robust_stats: ClassVar[bool] = True  # uses median/IQR instead of mean/std
    assumes_normality: ClassVar[bool] = False  # requires normal distribution
    sensitive_to_sample_size: ClassVar[bool] = False  # performs poorly on small samples
    min_sample_size: ClassVar[int] = 0

    @property
    def phase(self) -> TransformPhase:
        return TransformPhase.OUTLIER

    def should_apply(self, stats: NumericalColumnStats, meta: ColumnMeta) -> bool:
        if meta.is_target:
            return False
        if meta.semantic_role in (
            SemanticRole.ID_LIKE,
            SemanticRole.BINARY,
            SemanticRole.NUMERIC_NOMINAL,
        ):
            return False
        if self.assumes_normality and abs(stats.skewness) >= 1.0:
            return False
        if self.sensitive_to_sample_size and stats.count < self.min_sample_size:
            return False
        return stats.outlier_ratio > 0

    def explain(self, stats: NumericalColumnStats) -> str:
        parts = [f"Handling outliers with '{self.name}'"]
        parts.append(f"outlier_ratio={stats.outlier_ratio:.1%}")
        parts.append(f"skewness={stats.skewness:.2f}")
        if self.uses_robust_stats:
            parts.append("uses median/IQR (robust)")
        if self.assumes_normality:
            parts.append("assumes normality")
        if self.clips_values:
            parts.append("clips values")
        else:
            parts.append("removes values → sets NaN")
        return " | ".join(parts)


class IQRStrategy(NumericalOutlierStrategy):
    clips_values = True
    uses_robust_stats = True
    assumes_normality = False
    sensitive_to_sample_size = False

    def __init__(self) -> None:
        super().__init__(name="iqr")

    def should_apply(self, stats: NumericalColumnStats, meta: ColumnMeta) -> bool:
        return not meta.is_target and abs(stats.skewness) < 1.5

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        iqr = stats.q3 - stats.q1
        return data.clip(lower=stats.q1 - 1.5 * iqr, upper=stats.q3 + 1.5 * iqr)

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = _semantic_role_penalty(meta)

        if abs(stats.skewness) < 1.0:
            condition += 100  # IQR is symmetric — best for symmetric distributions

        if stats.outlier_ratio < 0.05:
            condition += 75  # few outliers — gentle clip is enough

        if abs(stats.skewness) >= 1.5:
            penalty += 100  # IQR fences become wrong for skewed distributions

        if meta.profile.is_heavy_tailed:
            penalty += (
                50  # heavy tails push IQR fences too far in → misses real outliers
            )

        if meta.profile.is_low_variance:
            penalty += (
                40  # very tight distribution → IQR is tiny, any deviation gets clipped
            )

        if meta.profile.is_sparse:
            penalty += 30

        if meta.semantic_role == SemanticRole.CONTINUOUS:
            condition += 50  # IQR is the natural first choice for continuous variables

        if meta.semantic_role == SemanticRole.COUNT:
            penalty += 30

        return DecisionScore(base=250, condition=condition, penalty=penalty)


class ZScoreStrategy(NumericalOutlierStrategy):
    clips_values = True
    uses_robust_stats = False  # mean/std are sensitive to outliers
    assumes_normality = True  # z-score is only meaningful with normality
    sensitive_to_sample_size = True  # std is unstable on small samples
    min_sample_size = 30

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold
        super().__init__(name="z-score")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        lower = stats.mean - self.threshold * stats.std
        upper = stats.mean + self.threshold * stats.std

        return data.clip(lower=lower, upper=upper)

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = _semantic_role_penalty(meta)

        if abs(stats.skewness) < 0.5:
            condition += 125  # z-score is most meaningful when near-normal

        if abs(stats.skewness) >= 1.0:
            penalty += 125  # skew distorts mean/std → thresholds become unreliable

        if stats.count >= 100:
            condition += 50  # std is stable on large samples

        if stats.count < 30:
            penalty += 100  # std unstable on small samples

        if meta.profile.is_heavy_tailed:
            penalty += (
                75  # heavy tails inflate std → threshold is too wide, misses outliers
            )

        if not meta.profile.is_skewed and not meta.profile.is_heavy_tailed:
            condition += 50  # well-behaved bell curve → z-score is ideal

        if meta.profile.is_sparse:
            penalty += 50  # mean/std computed over few values → unreliable thresholds

        if meta.semantic_role == SemanticRole.CONTINUOUS:
            condition += 40  # continuous features are the primary use-case for z-score

        if meta.semantic_role == SemanticRole.COUNT:
            penalty += 75  # counts follow Poisson-like distributions, not Gaussian

        if meta.semantic_role == SemanticRole.ORDINAL:
            penalty += (
                50  # ordinal ranks don't have a meaningful Gaussian interpretation
            )

        return DecisionScore(base=250, condition=condition, penalty=penalty)


class WinsorizeStrategy(NumericalOutlierStrategy):
    clips_values = True
    uses_robust_stats = True  # p05/p95 are percentiles, which are robust
    assumes_normality = False
    sensitive_to_sample_size = True  # percentiles are unstable on small samples
    min_sample_size = 20

    def __init__(self) -> None:
        super().__init__(name="winsorize")

    def should_apply(self, stats: NumericalColumnStats, meta: ColumnMeta) -> bool:
        return super().should_apply(stats, meta) and stats.outlier_ratio > 0.01

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.clip(stats.p05, stats.p95)

    def score(self, meta: ColumnMeta, stats: NumericalColumnStats) -> DecisionScore:
        condition = 0
        penalty = _semantic_role_penalty(meta)

        if abs(stats.skewness) >= 1.0:
            condition += 100  # percentile-based clipping adapts to skew naturally

        if stats.outlier_ratio >= 0.05:
            condition += 100  # many outliers → winsorize clips both tails by ratio

        if stats.count < 20:
            condition -= 75  # percentiles are unstable on very small samples

        if meta.profile.is_heavy_tailed:
            condition += 75  # heavy tails are exactly what winsorize is designed for

        if meta.profile.has_outliers and stats.outlier_ratio >= 0.03:
            condition += 50  # confirmed outlier-heavy columns benefit most

        if meta.profile.is_sparse:
            penalty += 40  # too few points → p05/p95 collapse toward the same value

        if meta.profile.is_low_variance:
            penalty += 30  # clipping a tight distribution risks removing valid signal

        if meta.semantic_role == SemanticRole.CONTINUOUS:
            condition += (
                30  # winsorize is a safe general-purpose choice for continuous data
            )

        if meta.semantic_role == SemanticRole.COUNT:
            condition += (
                50  # right-skewed counts benefit from one-sided percentile capping
            )

        return DecisionScore(base=250, condition=condition, penalty=penalty)


def _semantic_role_penalty(meta: ColumnMeta) -> int:
    """
    some semantic roles make outlier clipping
    actively harmful regardless of the chosen strategy.
    Returns a penalty to subtract from the score.
    """
    role = meta.semantic_role
    penalty = 0

    # IDs carry no real distribution — outlier treatment is meaningless
    if role == SemanticRole.ID_LIKE:
        penalty += 300

    # Binary columns have only two values — nothing to clip
    if role == SemanticRole.BINARY:
        penalty += 300

    # Ordinal columns have meaningful rank order; clipping distorts that
    if role == SemanticRole.ORDINAL:
        penalty += 75

    # Count data is non-negative and right-skewed by nature;
    # aggressive clipping removes valid extreme counts
    if role == SemanticRole.COUNT:
        penalty += 50

    if role == SemanticRole.NUMERIC_NOMINAL:
        penalty += 60

    return penalty
