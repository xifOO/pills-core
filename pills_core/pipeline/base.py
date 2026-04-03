from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Tuple

import pandas as pd

from pills_core._enums import TransformPhase
from pills_core.strategies.base import EmbeddingT, MetaT, SingleStrategy
from pills_core.types.stats import StatsT


@dataclass
class FittedColumnArtifact(Generic[StatsT, MetaT, EmbeddingT]):
    name: str
    meta: MetaT
    embedding: EmbeddingT
    phase_order: Tuple[TransformPhase, ...]
    strategies: Dict[TransformPhase, SingleStrategy]
    phase_stats: Dict[TransformPhase, StatsT]


class BasePipeline(Generic[StatsT, MetaT, EmbeddingT]):
    @abstractmethod
    def fit(self, series: pd.Series, is_target: bool) -> FittedColumnArtifact: ...

    @abstractmethod
    def transform(
        self, series: pd.Series, artifact: FittedColumnArtifact
    ) -> pd.Series: ...

    def fit_transform(
        self, series: pd.Series, is_target: bool
    ) -> Tuple[FittedColumnArtifact, pd.Series]:
        artifact = self.fit(series, is_target)
        result = self.transform(series, artifact)
        return artifact, result
