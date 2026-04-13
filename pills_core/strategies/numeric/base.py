from abc import ABC
from dataclasses import dataclass

from pills_core._enums import ColumnRole
from pills_core.strategies.base import ColumnMeta, SingleStrategy, StrategyEmbedding
from pills_core.strategies.util import register_required_stats
from pills_core.types.profiles import NumericalDomainProfile, StatisticalProfile
from pills_core.types.stats import NumericalColumnStats


@dataclass
class NumericalColumnMeta(ColumnMeta):
    domain_profile: NumericalDomainProfile
    profile: StatisticalProfile


@dataclass
class NumericalEmbedding(StrategyEmbedding):
    skewness_sensitivity: float  # how well it handles skewed distributions
    outliers_sensitivity: float  # how much outliers degrade it (higher = worse)


@register_required_stats("outlier_ratio", "skewness")
class NumericalStrategy(
    SingleStrategy[NumericalColumnStats, NumericalColumnMeta, NumericalEmbedding], ABC
):
    @property
    def column_type(self) -> ColumnRole:
        return ColumnRole.NUMERICAL

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        return True

    def is_task_valid(self, meta: NumericalColumnMeta) -> bool:
        return True
