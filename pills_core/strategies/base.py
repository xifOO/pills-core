from abc import ABC, abstractmethod
from typing import Generic

import pandas as pd

from pills_core._enums import ColumnRole
from pills_core.types.stats import StatsT


class TransformStrategy(ABC, Generic[StatsT]):
    @property
    @abstractmethod
    def column_type(self) -> ColumnRole: ...

    @abstractmethod
    def should_apply(self, stats: StatsT, is_target: bool) -> bool: ...

    @abstractmethod
    def apply(self, data: pd.Series, stats: StatsT) -> pd.Series: ...


class SingleStrategy(TransformStrategy[StatsT], Generic[StatsT]):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def priority(self, stats: StatsT) -> int: ...
