from dataclasses import dataclass
from typing import List, TypeVar, Union

Numeric = Union[float, int]
StatsT = TypeVar("StatsT", bound="BaseColumnStats")


@dataclass
class BaseColumnStats:
    count: int
    n_unique: int
    missing_ratio: float
    unique_ratio: float


@dataclass
class NumericalColumnStats(BaseColumnStats):
    max: Numeric
    min: Numeric
    mean: Numeric
    median: Numeric
    mode: Numeric
    std: Numeric
    variance: Numeric
    skewness: float
    kurtosis: float
    range: Numeric
    outlier_ratio: float
    q1: Numeric
    q3: Numeric
    p05: Numeric
    p95: Numeric
    is_integer_valued: bool
    monotonic_ratio: float
    cv: float  # std / mean
    zero_ratio: float


@dataclass
class CategoricalColumnStats(BaseColumnStats):
    most_frequent: str
    most_frequent_ratio: float
    rare_categories: List[str]
    rare_ratio: float
    entropy: float
    mode: str


@dataclass(frozen=True, slots=True)
class NumericalThresholds:
    id_unique_ratio: float = 0.95
    id_monotonic_ratio: float = 0.95
    binary_max_unique: int = 2
    low_unique_ratio: float = 0.15
    low_unique_abs: int = 20
    ordinal_max_skewness: float = 1.0
    count_skewness: float = 0.5
    skewed_threshold: float = 1.0
    heavy_tail_kurtosis: float = 3.0
    sparse_missing_ratio: float = 0.3
    low_variance_cv: float = 0.01


@dataclass(frozen=True, slots=True)
class CategoricalThresholds:
    binary_max_unique: int = 2
    low_cardinality_max: int = 10
    medium_cardinality_max: int = 100
    rare_ratio_threshold: float = 0.05
    dominant_ratio_threshold: float = 0.95
    sparse_missing_ratio: float = 0.3
