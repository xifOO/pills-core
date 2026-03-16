from abc import ABC, abstractmethod
from typing import Final, Generic

import pandas as pd

from pills_core._enums import ColumnRole, SemanticRole
from pills_core.strategies.base import ColumnMeta, StrategyEmbedding
from pills_core.types.profiles import DomainProfile, StatisticalProfile
from pills_core.types.stats import NumericalColumnStats, StatsT


class ColumnAnalyzer(ABC, Generic[StatsT]):
    @abstractmethod
    def detect_column_role(self, series: pd.Series) -> ColumnRole: ...

    @abstractmethod
    def detect_semantic_role(self, stats: StatsT) -> SemanticRole: ...

    @abstractmethod
    def build_statistical_profile(self, stats: StatsT) -> StatisticalProfile: ...

    @abstractmethod
    def build_column_embedding(
        self, stats: StatsT, meta: ColumnMeta
    ) -> StrategyEmbedding: ...

    @abstractmethod
    def build_domain_profile(
        self, column_name: str, stats: StatsT
    ) -> DomainProfile: ...


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

    _DOMAIN_KEYWORDS: Final = {
        "monetary": {
            "amnt",
            "amount",
            "bal",
            "balance",
            "limit",
            "installment",
            "recoveries",
            "fee",
            "payment",
            "pymnt",
            "prncp",
            "inv",
        },
        "rate": {
            "int_rate",
            "revol_util",
            "il_util",
            "bc_util",
            "all_util",
            "sec_app_revol_util",
            "pct",
            "percent",
            "dti",
            "rate",
        },
        "ratio": {"dti", "util", "pct", "percent"},
        "score": {"fico", "grade"},
        "count": {
            "acc",
            "inq",
            "delinq",
            "rec",
            "tl",
            "mths",
            "mo_sin",
            "num_",
            "open_",
            "pub_rec",
            "collections",
            "bankruptcies",
            "liens",
            "term",
        },
    }

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

    def build_column_embedding(
        self,
        stats: NumericalColumnStats,
        meta: ColumnMeta,
    ) -> StrategyEmbedding:
        skewness_sensitivity = min(abs(stats.skewness) / 3.0, 1.0)
        outliers_sensitivity = min(stats.outlier_ratio * 5.0, 1.0)

        distribution_preservation = 1.0
        if meta.profile.is_skewed:
            distribution_preservation -= 0.4
        if meta.profile.is_heavy_tailed:
            distribution_preservation -= 0.3
        distribution_preservation = max(distribution_preservation, 0.0)

        target_safety = 1.0 if meta.is_target else 0.0
        cardinality_fit = 1.0 - min(stats.unique_ratio, 1.0)

        return StrategyEmbedding(
            skewness_sensitivity=skewness_sensitivity,
            outliers_sensitivity=outliers_sensitivity,
            missing_ratio_fit=1.0 if stats.missing_ratio > 0 else 0.0,
            distribution_preservation=distribution_preservation,
            target_safety=target_safety,
            cardinality_fit=cardinality_fit,
        )

    def build_domain_profile(
        self,
        column_name: str,
        stats: NumericalColumnStats,
    ) -> DomainProfile:
        name = column_name.lower()

        is_monetary = self._matches(name, self._DOMAIN_KEYWORDS["monetary"])
        is_rate = self._matches(name, self._DOMAIN_KEYWORDS["rate"])
        is_ratio = self._matches(name, self._DOMAIN_KEYWORDS["ratio"])
        is_score = self._matches(name, self._DOMAIN_KEYWORDS["score"])

        is_stat_bounded = (
            stats.min >= 0 and stats.max <= 1.0 and not is_monetary and not is_score
        )
        is_bounded = is_rate or is_ratio or is_score or is_stat_bounded

        lower_bound = 0.0 if is_monetary or is_bounded else None

        if is_rate or is_ratio:
            upper_bound = 100.0 if stats.max > 1.0 else 1.0
        else:
            upper_bound = None

        return DomainProfile(
            is_ratio=is_ratio,
            is_monetary=is_monetary,
            is_rate=is_rate,
            is_bounded=is_bounded,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def _matches(self, name: str, keywords: set[str]) -> bool:
        return any(kw in name for kw in keywords)
