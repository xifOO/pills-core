from abc import ABC, abstractmethod
from typing import Generic
import pandas as pd
from pills_core._enums import ColumnRole, SemanticRole
from pills_core.types.profiles import StatisticalProfile
from pills_core.types.stats import NumericalColumnStats, StatsT


class ColumnAnalyzer(ABC, Generic[StatsT]):
    @abstractmethod
    def detect_column_role(self, series: pd.Series) -> ColumnRole: ...

    @abstractmethod
    def detect_semantic_role(self, stats: StatsT) -> SemanticRole: ...

    @abstractmethod
    def build_statistical_profile(self, stats: StatsT) -> StatisticalProfile: ...


class NumericalColumnAnalyzer(ColumnAnalyzer[NumericalColumnStats]):
    ID_UNIQUE_RATIO_THRESHOLD = 0.95
    ID_MONOTONIC_THRESHOLD = 0.95
    BINARY_MAX_UNIQUE = 2
    LOW_UNIQUE_RATIO_THRESHOLD = 0.15
    LOW_UNIQUE_ABS_THRESHOLD = 20
    ORDINAL_MAX_SKEWNESS = 1.0
    COUNT_SKEWNESS_THRESHOLD = 0.5
    SKEWED_THRESHOLD = 1.0
    HEAVY_TAIL_KURTOSIS = 3.0
    SPARSE_MISSING_RATIO = 0.3
    LOW_VARIANCE_CV_THRESHOLD = 0.01

    def detect_column_role(self, series: pd.Series) -> ColumnRole:
        if pd.api.types.is_numeric_dtype(series):
            return ColumnRole.NUMERICAL
        return ColumnRole.DROP

    def detect_semantic_role(
        self,
        stats: NumericalColumnStats,
    ) -> SemanticRole:
        if stats.n_unique <= self.BINARY_MAX_UNIQUE:
            return SemanticRole.BINARY

        if (
            stats.unique_ratio >= self.ID_UNIQUE_RATIO_THRESHOLD
            and stats.monotonic_ratio >= self.ID_MONOTONIC_THRESHOLD
        ):
            return SemanticRole.ID_LIKE

        if (
            stats.is_integer_valued
            and stats.min >= 0
            and stats.skewness >= self.COUNT_SKEWNESS_THRESHOLD
            and stats.zero_ratio < 0.95
        ):
            return SemanticRole.COUNT

        if (
            stats.is_integer_valued
            and stats.min >= 0
            and stats.zero_ratio >= 0.95
            and stats.n_unique <= 10
        ):
            return SemanticRole.COUNT

        is_low_unique = (
            stats.unique_ratio <= self.LOW_UNIQUE_RATIO_THRESHOLD
            and stats.n_unique <= self.LOW_UNIQUE_ABS_THRESHOLD
        )

        if is_low_unique and stats.is_integer_valued:
            if abs(stats.skewness) <= self.ORDINAL_MAX_SKEWNESS:
                return SemanticRole.ORDINAL
            return SemanticRole.NUMERIC_NOMINAL

        return SemanticRole.CONTINUOUS

    def build_statistical_profile(
        self,
        stats: NumericalColumnStats,
    ) -> StatisticalProfile:
        return StatisticalProfile(
            is_skewed=bool(abs(stats.skewness) >= self.SKEWED_THRESHOLD),
            is_heavy_tailed=bool(stats.kurtosis > self.HEAVY_TAIL_KURTOSIS),
            has_outliers=stats.outlier_ratio > 0,
            is_sparse=stats.missing_ratio >= self.SPARSE_MISSING_RATIO,
            is_low_variance=stats.cv < self.LOW_VARIANCE_CV_THRESHOLD,
        )
