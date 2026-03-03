from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import pandas as pd

from pills_core._enums import TransformPhase


T = TypeVar("T", pd.Series, pd.DataFrame)


@dataclass
class TransformResult:
    data: pd.Series
    strategy_name: str
    phase: TransformPhase
    before_stats: dict = field(default_factory=dict)
    after_stats: dict = field(default_factory=dict)
    notes: Optional[str] = None


class Descriptor(ABC, Generic[T]):
    """Base descriptor defining the contract for all data transformations"""

    priority: int

    @abstractmethod
    def match(self, data: T) -> bool:
        """Check if this descriptor is applicable to the data"""
        ...

    @abstractmethod
    def transform(self, data: T) -> T:
        """Apply the descriptor's transformation to the data"""
        ...

    def explain(self, data: T) -> str:
        """Return a readable explanation"""
        ...
