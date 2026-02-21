from dataclasses import dataclass, field
from enum import Enum, auto


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
