from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pills_core._enums import TransformPhase
from pills_core.strategies.base import SingleStrategy
from pills_core.types.stats import BaseColumnStats


@dataclass(frozen=True, slots=True)
class Step:
    phase: TransformPhase
    strategy: SingleStrategy
    stats: BaseColumnStats

    def apply(self, data: pd.Series) -> pd.Series:
        return self.strategy.apply(data, self.stats)

    @property
    def name(self) -> str:
        return self.strategy.name

    def __repr__(self) -> str:
        return f"Step({self.phase.value}:{self.name})"
