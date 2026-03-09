from dataclasses import dataclass
from typing import List, TypeVar, Union

Numeric = Union[float, int]

StatsT = TypeVar("StatsT")


@dataclass
class NumericalColumnStats:
    max: Numeric
    min: Numeric
    mean: Numeric
    median: Numeric
    mode: Numeric
    std: Numeric
    count: int
    variance: Numeric
    skewness: float
    kurtosis: float
    range: Numeric
    n_unique: Numeric
    missing_ratio: float
    outlier_ratio: float
    q1: Numeric
    q3: Numeric
    p05: Numeric
    p95: Numeric
    is_integer_valued: bool
    monotonic_ratio: float
    cv: float  # std / mean
    unique_ratio: float
    zero_ratio: float


@dataclass
class CategoricalColumnStats:
    n_unique: Numeric
    missing_ratio: float
    most_frequent: str
    most_frequent_ratio: float
    rare_categories: List[str]
    rare_ratio: float
    entropy: float
    mode: str
