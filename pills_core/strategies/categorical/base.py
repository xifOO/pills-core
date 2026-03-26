from abc import ABC
from dataclasses import dataclass

from pills_core._enums import ColumnRole
from pills_core.strategies.base import ColumnMeta, SingleStrategy
from pills_core.types.profiles import CategoricalProfile
from pills_core.types.stats import CategoricalColumnStats


@dataclass
class CategoricalColumnMeta(ColumnMeta):
    profile: CategoricalProfile


class CategoricalStrategy(
    SingleStrategy[CategoricalColumnStats, CategoricalColumnMeta], ABC
):
    @property
    def column_type(self) -> ColumnRole:
        return ColumnRole.CATEGORICAL

    def is_domain_valid(self, meta: CategoricalColumnMeta) -> bool:
        return True

    def is_task_valid(self, meta: CategoricalColumnMeta) -> bool:
        return True
