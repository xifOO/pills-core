from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Optional

import pandas as pd

from pills_core._enums import ColumnRole, TransformPhase
from pills_core.types.stats import StatsT


@dataclass
class TransformResult:
    data: pd.Series
    strategy_name: str
    phase: TransformPhase
    before_stats: dict = field(default_factory=dict)
    after_stats: dict = field(default_factory=dict)
    notes: Optional[str] = None


class TransformStrategy(ABC, Generic[StatsT]):
    @property
    @abstractmethod
    def column_type(self) -> ColumnRole: ...

    @property
    @abstractmethod
    def phase(self) -> TransformPhase: ...

    @abstractmethod
    def should_apply(self, stats: StatsT, is_target: bool) -> bool: ...

    @abstractmethod
    def apply(self, data: pd.Series, stats: StatsT) -> pd.Series: ...

    def explain(self, stats: StatsT) -> str:
        return ""
    

class SingleStrategy(TransformStrategy[StatsT], Generic[StatsT]):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def priority(self, stats: StatsT) -> int: ...
