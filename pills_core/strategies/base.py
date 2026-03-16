from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import ClassVar, Dict, Generic, Optional

import numpy as np
import pandas as pd

from pills_core._enums import ColumnRole, FamilyRole, SemanticRole, TransformPhase
from pills_core.types.profiles import DomainProfile, StatisticalProfile
from pills_core.types.stats import StatsT


@dataclass
class ColumnMeta:
    role: ColumnRole
    semantic_role: SemanticRole
    profile: StatisticalProfile
    is_target: bool
    domain_profile: DomainProfile


@dataclass
class StrategyEmbedding:
    """
    Fixed-length semantic vector describing a strategy's behavior.
    Used for similarity search / nearest-neighbor strategy retrieval.

    All values are in [0.0, 1.0] unless noted otherwise.
    """

    skewness_sensitivity: float  # how well it handles skewed distributions
    outliers_sensitivity: float  # how much outliers degrade it (higher = worse)
    missing_ratio_fit: float  # how well it handles high missing ratios
    distribution_preservation: float  # how much it preverses the original shape
    target_safety: float  # safe to apply on target variable
    cardinality_fit: float  # fit for low-cardinality columns

    def to_weighted_array(self, weights_dict: dict[str, float]) -> np.ndarray:
        data = asdict(self)
        weighted_values = [data[field] * weights_dict.get(field, 1.0) for field in data]
        return np.array(weighted_values, dtype=np.float32)


class TransformStrategy(ABC, Generic[StatsT]):
    @property
    @abstractmethod
    def column_type(self) -> ColumnRole: ...

    @property
    @abstractmethod
    def phase(self) -> TransformPhase: ...

    @abstractmethod
    def is_domain_valid(self, meta: ColumnMeta) -> bool: ...

    @abstractmethod
    def should_apply(self, stats: StatsT, meta: ColumnMeta) -> bool: ...

    @abstractmethod
    def apply(self, data: pd.Series, stats: StatsT) -> pd.Series: ...

    def explain(self, stats: StatsT) -> str:
        return ""


class SingleStrategy(TransformStrategy[StatsT], Generic[StatsT]):
    family_role: ClassVar[FamilyRole]

    embedding: ClassVar[StrategyEmbedding]
    radius: ClassVar[float] = 1.0

    def __init__(self, name: str) -> None:
        self.name = name

    def is_domain_valid(self, meta: ColumnMeta) -> bool:
        return True

    def distance(
        self, column_embedding: StrategyEmbedding, weights: Dict[str, float]
    ) -> float:
        return float(
            np.linalg.norm(
                self.embedding.to_weighted_array(weights)
                - column_embedding.to_weighted_array(weights)
            )
        )

    def score(
        self,
        column_embedding: StrategyEmbedding,
        stats: StatsT,
        meta: ColumnMeta,
        weights: Dict[str, float],
    ) -> Optional[float]:

        if not self.should_apply(stats, meta):
            return None

        if not self.is_domain_valid(meta):
            return None

        dist = self.distance(column_embedding, weights)

        if dist > self.radius:
            return None

        return dist
