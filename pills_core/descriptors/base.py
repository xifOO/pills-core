from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pandas as pd


T = TypeVar("T", pd.Series, pd.DataFrame)


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
