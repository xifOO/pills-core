from dataclasses import dataclass
from typing import List, Union, TypeVar


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
