from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import pandas as pd

from pills_core.pipeline.step import Step


@dataclass
class TransformSequence:
    steps: Tuple[Step, ...]

    def apply(self, series: pd.Series) -> pd.Series:
        result = series.copy()
        for step in self.steps:
            result = step.apply(result)
        return result

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        chain = " → ".join(f"{step.phase.value}:{step.name}" for step in self.steps)
        return f"TransformSequence({chain})"
