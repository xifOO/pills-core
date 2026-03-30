from abc import ABC
from dataclasses import dataclass

from pills_core._enums import ColumnRole
from pills_core.strategies.base import ColumnMeta, SingleStrategy, StrategyEmbedding
from pills_core.types.profiles import CategoricalProfile
from pills_core.types.stats import CategoricalColumnStats


@dataclass
class CategoricalColumnMeta(ColumnMeta):
    profile: CategoricalProfile


@dataclass
class CategoricalEmbedding(StrategyEmbedding):
    rare_categories_handling: (
        float  # how well it handles rare/infrequent categories (1.0 = excellent)
    )
    imbalance_sensitivity: (
        float  # sensitivity to class imbalance (1.0 = very sensitive, worse)
    )
    order_awareness: (
        float  # whether it respects ordinal relationships (1.0 = ordinal-aware)
    )
    typo_tolerance: float  # requires typo cleaning beforehand (1.0 = requires cleaning)


class CategoricalStrategy(
    SingleStrategy[CategoricalColumnStats, CategoricalColumnMeta, CategoricalEmbedding],
    ABC,
):
    @property
    def column_type(self) -> ColumnRole:
        return ColumnRole.CATEGORICAL

    def is_domain_valid(self, meta: CategoricalColumnMeta) -> bool:
        return True

    def is_task_valid(self, meta: CategoricalColumnMeta) -> bool:
        return True
