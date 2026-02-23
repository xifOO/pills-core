import pandas as pd

from pills_core._enums import TransformPhase
from pills_core.strategies.numeric.base import NumericalStrategy
from pills_core.strategies.priorities import for_iqr, for_winsorize, for_zscore
from pills_core.types.stats import NumericalColumnStats


class NumericalOutlierStrategy(NumericalStrategy):
    clips_values: bool = True # strategy clips or removes values
    uses_robust_stats: bool = True # uses median/IQR instead of mean/std
    assumes_normality: bool = False # requires normal distribution
    sensitive_to_sample_size: bool = False # performs poorly on small samples
    min_sample_size: int = 0

    @property
    def phase(self) -> TransformPhase:
        return TransformPhase.OUTLIER

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        if is_target:
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

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        return not is_target and abs(stats.skewness) < 1.5

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        iqr = stats.q3 - stats.q1
        return data.clip(lower=stats.q1 - 1.5 * iqr, upper=stats.q3 + 1.5 * iqr)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_iqr(stats))


class ZScoreStrategy(NumericalOutlierStrategy):
    clips_values = True
    uses_robust_stats = False     # mean/std are sensitive to outliers
    assumes_normality = True      # z-score is only meaningful with normality
    sensitive_to_sample_size = True  # std is unstable on small samples
    min_sample_size = 30 

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold
        super().__init__(name="z-score")

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        lower = stats.mean - self.threshold * stats.std
        upper = stats.mean + self.threshold * stats.std

        return data.clip(lower=lower, upper=upper)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_zscore(stats))


class WinsorizeStrategy(NumericalOutlierStrategy):
    clips_values = True
    uses_robust_stats = True # p05/p95 are percentiles, which are robust
    assumes_normality = False
    sensitive_to_sample_size = True  # percentiles are unstable on small samples
    min_sample_size = 20

    def __init__(self) -> None:
        super().__init__(name="winsorize")

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        return super().should_apply(stats, is_target) and stats.outlier_ratio > 0.01

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.clip(stats.p05, stats.p95)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_winsorize(stats))
