import numpy as np
import pandas as pd

from pills_core._enums import TransformPhase
from pills_core.strategies.numeric.base import NumericalStrategy
from pills_core.strategies.priorities import for_log_transform, for_minmax_scaler, for_robust_scaler, for_standard_scaler
from pills_core.types.stats import NumericalColumnStats


class NumericalScalingStrategy(NumericalStrategy):
    requires_non_negative: bool = False # LogTransform, SqrtTransform
    is_invertible: bool = True # whether inverse denormalization is possible
    preserves_distribution: bool = True # StandardScaler/Robust; False for Log
    sensitive_to_outliers: bool = False
    assumes_normality: bool = False

    @property 
    def phase(self) -> TransformPhase:
        return TransformPhase.SCALING

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        if is_target:
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
    
    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_standard_scaler(stats))
    

class MinMaxScalerStrategy(NumericalScalingStrategy):
    sensitive_to_outliers = True   # → should_apply will block if outlier_ratio >= 0.05
    assumes_normality = False
    preserves_distribution = True
    is_invertible = True

    def __init__(self) -> None:
        super().__init__(name="min_max_scaler")

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        return super().should_apply(stats, is_target) and (stats.max - stats.min) > 0

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return (data - stats.min) / (stats.max - stats.min)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_minmax_scaler(stats))


class LogTransformStrategy(NumericalScalingStrategy):
    requires_non_negative = True   # → should_apply will block if min < 0
    preserves_distribution = False
    sensitive_to_outliers = False
    is_invertible = True

    def __init__(self):
        super().__init__(name="log_transform")

    def should_apply(self, stats: NumericalColumnStats, is_target: bool) -> bool:
        return super().should_apply(stats, is_target) and stats.skewness > 1.5

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return pd.Series(np.log1p(data), index=data.index)

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_log_transform(stats))


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

    def priority(self, stats: NumericalColumnStats) -> int:
        return int(for_robust_scaler(stats))