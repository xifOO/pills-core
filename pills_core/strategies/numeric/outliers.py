from typing import ClassVar

import pandas as pd

from pills_core._enums import FamilyRole, SemanticRole, TransformPhase
from pills_core.strategies.base import StrategyEmbedding
from pills_core.strategies.numeric.base import NumericalColumnMeta, NumericalStrategy
from pills_core.types.stats import NumericalColumnStats


class NumericalOutlierStrategy(NumericalStrategy):
    clips_values: ClassVar[bool] = True  # strategy clips or removes values
    uses_robust_stats: ClassVar[bool] = True  # uses median/IQR instead of mean/std
    assumes_normality: ClassVar[bool] = False  # requires normal distribution
    sensitive_to_sample_size: ClassVar[bool] = False  # performs poorly on small samples
    min_sample_size: ClassVar[int] = 0
    requires_imputed: ClassVar[bool] = True

    @property
    def phase(self) -> TransformPhase:
        return TransformPhase.OUTLIER

    def should_apply(
        self, stats: NumericalColumnStats, meta: NumericalColumnMeta
    ) -> bool:
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
    name: ClassVar[str] = "iqr"
    family_role: ClassVar[FamilyRole] = FamilyRole.ROBUST

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.4,
        outliers_sensitivity=0.1,
        missing_ratio_fit=0.5,
        distribution_preservation=0.7,
        target_safety=0.0,
        cardinality_fit=0.4,
    )
    radius = 1.4

    clips_values = True
    uses_robust_stats = True
    assumes_normality = False
    sensitive_to_sample_size = False

    def should_apply(
        self, stats: NumericalColumnStats, meta: NumericalColumnMeta
    ) -> bool:
        return (
            not meta.is_target and abs(stats.skewness) < 1.5 and stats.outlier_ratio > 0
        )

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        if (
            meta.domain_profile.is_bounded
            and meta.domain_profile.upper_bound is not None
        ):
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        iqr = stats.q3 - stats.q1
        return data.clip(lower=stats.q1 - 1.5 * iqr, upper=stats.q3 + 1.5 * iqr)


class ZScoreStrategy(NumericalOutlierStrategy):
    name: ClassVar[str] = "z-score"
    family_role: ClassVar[FamilyRole] = FamilyRole.STATISTICAL

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.1,
        outliers_sensitivity=0.8,
        missing_ratio_fit=0.5,
        distribution_preservation=0.7,
        target_safety=0.0,
        cardinality_fit=0.3,
    )
    radius = 1.1

    clips_values = True
    uses_robust_stats = False  # mean/std are sensitive to outliers
    assumes_normality = True  # z-score is only meaningful with normality
    sensitive_to_sample_size = True  # std is unstable on small samples
    min_sample_size = 30

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        lower = stats.mean - self.threshold * stats.std
        upper = stats.mean + self.threshold * stats.std

        return data.clip(lower=lower, upper=upper)


class WinsorizeStrategy(NumericalOutlierStrategy):
    name: ClassVar[str] = "winsorize"
    family_role: ClassVar[FamilyRole] = FamilyRole.PERCENTILE

    embedding = StrategyEmbedding(
        skewness_sensitivity=0.8,
        outliers_sensitivity=0.2,
        missing_ratio_fit=0.5,
        distribution_preservation=0.6,
        target_safety=0.0,
        cardinality_fit=0.3,
    )
    radius = 1.6

    clips_values = True
    uses_robust_stats = True  # p05/p95 are percentiles, which are robust
    assumes_normality = False
    sensitive_to_sample_size = True  # percentiles are unstable on small samples
    min_sample_size = 20

    def should_apply(
        self, stats: NumericalColumnStats, meta: NumericalColumnMeta
    ) -> bool:
        return super().should_apply(stats, meta) and stats.outlier_ratio > 0.01

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        if meta.is_target and meta.domain_profile.is_bounded:
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.clip(stats.p05, stats.p95)
