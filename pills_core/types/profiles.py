from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


class Cardinality(Enum):
    LOW = auto()  # 2-10
    MEDIUM = auto()  # 10-100
    HIGH = auto()  # 100+


@dataclass
class CategoricalProfile:
    """Everything Inspector learned about the column."""

    cardinality: Cardinality
    n_unique: int
    rare_categories: list[str] = field(default_factory=list)
    has_typos: bool = False
    has_order: bool = False  # low/medium/high
    is_domain_specific: bool = False


@dataclass
class StatisticalProfile:
    is_skewed: bool
    is_heavy_tailed: bool
    has_outliers: bool
    is_sparse: bool
    is_low_variance: bool


@dataclass
class DomainProfile:
    is_ratio: bool
    is_monetary: bool
    is_rate: bool
    is_score: bool
    is_bounded: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass
class CategoricalDomainProfile:
    encoding_scheme: str
    expected_categories: List[str] = field(default_factory=list)
    is_encoded: bool = False
    is_sensitive: bool = False
