from dataclasses import dataclass
from typing import Dict, List, Union


Numeric = Union[float, int]


@dataclass
class NumericalColumnStats:
    max: Numeric
    min: Numeric
    mean: Numeric
    median: Numeric
    std: Numeric
    variance: Numeric
    skewness: float
    range: Numeric
    n_unique: Numeric
    quantiles: Dict[str, Numeric]


@dataclass
class CategoricalColumnStats:
    n_unique: Numeric
    missing_ratio: float
    most_frequent: str
    most_frequent_ratio: float
    rare_categories: List[str]
    rare_ratio: float
    entropy: float
    has_typos: bool
