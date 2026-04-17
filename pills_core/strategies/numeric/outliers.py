from typing import ClassVar

import pandas as pd

from pills_core._enums import FamilyRole, SemanticRole, TransformPhase
from pills_core.explain import Explanation
from pills_core.strategies.numeric.base import (
    NumericalColumnMeta,
    NumericalEmbedding,
    NumericalStrategy,
)
from pills_core.types.stats import NumericalColumnStats


class NumericalOutlierStrategy(NumericalStrategy):
    clips_values: ClassVar[bool] = True  # strategy clips or removes values
    uses_robust_stats: ClassVar[bool] = True  # uses median/IQR instead of mean/std
    assumes_normality: ClassVar[bool] = False  # requires normal distribution
    sensitive_to_sample_size: ClassVar[bool] = False  # performs poorly on small samples
    requires_imputed: ClassVar[bool] = True

    def __init__(
        self,
        *,
        embedding: NumericalEmbedding,
        radius: float,
        normality_skewness_limit: float = 1.0,
        min_sample_size: int = 30,
    ) -> None:
        super().__init__(embedding=embedding, radius=radius)
        self.normality_skewness_limit = normality_skewness_limit
        self.min_sample_size = min_sample_size

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
        if (
            self.assumes_normality
            and abs(stats.skewness) >= self.normality_skewness_limit
        ):
            return False
        if self.sensitive_to_sample_size and stats.count < self.min_sample_size:
            return False
        return stats.outlier_ratio > 0

    def explain(
        self, stats: NumericalColumnStats, meta: NumericalColumnMeta
    ) -> Explanation:
        reasons = [f"Handling outliers with '{self.name}'"]
        reasons.append(f"outlier_ratio={stats.outlier_ratio:.1%}")
        reasons.append(f"skewness={stats.skewness:.2f}")
        if self.uses_robust_stats:
            reasons.append("uses median/IQR (robust)")
        if self.assumes_normality:
            reasons.append("assumes normality")
        if self.clips_values:
            reasons.append("clips values")
        else:
            reasons.append("removes values → sets NaN")

        return Explanation(
            name=self.name,
            value="selected" if self.should_apply(stats, meta) else "rejected",
            reasons=reasons,
        )


class IQRStrategy(NumericalOutlierStrategy):
    name: ClassVar[str] = "iqr"
    family_role: ClassVar[FamilyRole] = FamilyRole.ROBUST

    clips_values: ClassVar[bool] = True
    uses_robust_stats: ClassVar[bool] = True
    assumes_normality: ClassVar[bool] = False
    sensitive_to_sample_size: ClassVar[bool] = False

    def __init__(
        self,
        *,
        embedding: NumericalEmbedding,
        radius: float,
        max_abs_skewness: float,
        min_outlier_ratio: float,
        clip_multiplier: float,
    ) -> None:
        super().__init__(embedding=embedding, radius=radius)
        self.max_abs_skewness = max_abs_skewness
        self.min_outlier_ratio = min_outlier_ratio
        self.clip_multiplier = clip_multiplier

    def should_apply(
        self, stats: NumericalColumnStats, meta: NumericalColumnMeta
    ) -> bool:
        return (
            not meta.is_target
            and abs(stats.skewness) < self.max_abs_skewness
            and stats.outlier_ratio > self.min_outlier_ratio
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
        return data.clip(
            lower=stats.q1 - self.clip_multiplier * iqr,
            upper=stats.q3 + self.clip_multiplier * iqr,
        )


class ZScoreStrategy(NumericalOutlierStrategy):
    name: ClassVar[str] = "z-score"
    family_role: ClassVar[FamilyRole] = FamilyRole.STATISTICAL

    clips_values: ClassVar[bool] = True
    uses_robust_stats: ClassVar[bool] = False  # mean/std are sensitive to outliers
    assumes_normality: ClassVar[bool] = (
        True  # z-score is only meaningful with normality
    )
    sensitive_to_sample_size: ClassVar[bool] = True  # std is unstable on small samples

    def __init__(
        self,
        *,
        embedding: NumericalEmbedding,
        radius: float,
        threshold: float,
        max_abs_skewness: float,
        min_sample_size: int,
    ) -> None:
        super().__init__(
            embedding=embedding,
            radius=radius,
            min_sample_size=min_sample_size,
            normality_skewness_limit=max_abs_skewness,
        )
        self.threshold = threshold

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        lower = stats.mean - self.threshold * stats.std
        upper = stats.mean + self.threshold * stats.std

        return data.clip(lower=lower, upper=upper)


class WinsorizeStrategy(NumericalOutlierStrategy):
    name: ClassVar[str] = "winsorize"
    family_role: ClassVar[FamilyRole] = FamilyRole.PERCENTILE

    clips_values: ClassVar[bool] = True
    uses_robust_stats: ClassVar[bool] = (
        True  # p05/p95 are percentiles, which are robust
    )
    assumes_normality: ClassVar[bool] = False
    sensitive_to_sample_size: ClassVar[bool] = (
        True  # percentiles are unstable on small samples
    )

    def __init__(
        self,
        *,
        embedding: NumericalEmbedding,
        radius: float,
        min_outlier_ratio: float,
        min_sample_size: int,
        lower_quantile: float,
        upper_quantile: float,
    ) -> None:
        super().__init__(
            embedding=embedding, radius=radius, min_sample_size=min_sample_size
        )
        self.min_outlier_ratio = min_outlier_ratio
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def should_apply(
        self, stats: NumericalColumnStats, meta: NumericalColumnMeta
    ) -> bool:
        return (
            super().should_apply(stats, meta)
            and stats.outlier_ratio > self.min_outlier_ratio
        )

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        if meta.is_target and meta.domain_profile.is_bounded:
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        lower = data.quantile(self.lower_quantile)
        upper = data.quantile(self.upper_quantile)
        return data.clip(lower, upper)
