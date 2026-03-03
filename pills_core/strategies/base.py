from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

import pandas as pd

from pills_core._enums import ColumnRole, SemanticRole, TransformPhase
from pills_core.strategies.score import DecisionScore
from pills_core.types.profiles import StatisticalProfile
from pills_core.types.stats import StatsT


@dataclass
class ColumnMeta:
    role: ColumnRole
    semantic_role: SemanticRole
    profile: StatisticalProfile
    is_target: bool


class TransformStrategy(ABC, Generic[StatsT]):
    @property
    @abstractmethod
    def column_type(self) -> ColumnRole: ...

    @property
    @abstractmethod
    def phase(self) -> TransformPhase: ...

    @abstractmethod
    def should_apply(self, stats: StatsT, meta: ColumnMeta) -> bool: ...

    @abstractmethod
    def apply(self, data: pd.Series, stats: StatsT) -> pd.Series: ...

    def explain(self, stats: StatsT) -> str:
        return ""


class SingleStrategy(TransformStrategy[StatsT], Generic[StatsT]):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def score(self, meta: ColumnMeta, stats: StatsT) -> DecisionScore: ...
