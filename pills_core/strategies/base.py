from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import ClassVar, Dict, Generic, Optional, Set, TypeVar

import numpy as np
import pandas as pd

from pills_core._enums import (
    ColumnRole,
    FamilyRole,
    SemanticRole,
    TaskType,
    TransformPhase,
)
from pills_core.types.stats import StatsT

MetaT = TypeVar("MetaT", bound="ColumnMeta")
EmbeddingT = TypeVar("EmbeddingT", bound="StrategyEmbedding")


@dataclass
class ColumnMeta:
    role: ColumnRole
    semantic_role: SemanticRole
    is_target: bool
    task_type: TaskType


@dataclass
class StrategyEmbedding:
    """
    Fixed-length semantic vector describing a strategy's behavior.
    Used for similarity search / nearest-neighbor strategy retrieval.

    All values are in [0.0, 1.0] unless noted otherwise.
    """

    missing_ratio_fit: float  # how well it handles high missing ratios
    distribution_preservation: float  # how much it preverses the original shape
    target_safety: float  # safe to apply on target variable
    cardinality_fit: float  # fit for low-cardinality columns

    def to_weighted_array(self, weights_dict: dict[str, float]) -> np.ndarray:
        data = asdict(self)
        weighted_values = [data[field] * weights_dict.get(field, 1.0) for field in data]
        return np.array(weighted_values, dtype=np.float32)


class TransformStrategy(ABC, Generic[StatsT, MetaT, EmbeddingT]):
    required_stats: ClassVar[Set[str]] = set()

    @property
    @abstractmethod
    def column_type(self) -> ColumnRole: ...

    @property
    @abstractmethod
    def phase(self) -> TransformPhase: ...

    @abstractmethod
    def is_domain_valid(self, meta: MetaT) -> bool: ...

    @abstractmethod
    def is_task_valid(self, meta: MetaT) -> bool: ...

    @abstractmethod
    def should_apply(self, stats: StatsT, meta: MetaT) -> bool: ...

    @abstractmethod
    def apply(self, data: pd.Series, stats: StatsT) -> pd.Series: ...

    def explain(self, stats: StatsT) -> str:
        return ""


class SingleStrategy(
    TransformStrategy[StatsT, MetaT, EmbeddingT], Generic[StatsT, MetaT, EmbeddingT]
):
    name: ClassVar[str]
    family_role: ClassVar[FamilyRole]

    def __init__(self, *, embedding: EmbeddingT, radius: float) -> None:
        self.embedding = embedding
        self.radius = radius

    def distance(
        self, column_embedding: EmbeddingT, weights: Dict[str, float]
    ) -> float:
        return float(
            np.linalg.norm(
                self.embedding.to_weighted_array(weights)
                - column_embedding.to_weighted_array(weights)
            )
        )

    def score(
        self,
        column_embedding: EmbeddingT,
        stats: StatsT,
        meta: MetaT,
        weights: Dict[str, float],
    ) -> Optional[float]:

        if not self.should_apply(stats, meta):
            return None

        if not self.is_domain_valid(meta):
            return None

        if not self.is_task_valid(meta):
            return None

        dist = self.distance(column_embedding, weights)

        if dist > self.radius:
            return None

        return dist

    def ordering_constraints(
        self, present_phases: set[TransformPhase]
    ) -> set[tuple[TransformPhase, TransformPhase]]:
        return set()
