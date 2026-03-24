from abc import ABC

from pills_core._enums import ColumnRole
from pills_core.strategies.base import ColumnMeta, SingleStrategy
from pills_core.types.stats import CategoricalColumnStats


class CategoricalStrategy(SingleStrategy[CategoricalColumnStats], ABC):
    @property
    def column_type(self) -> ColumnRole:
        return ColumnRole.CATEGORICAL

    def is_domain_valid(self, meta: ColumnMeta) -> bool:
        return True

    def is_task_valid(self, meta: ColumnMeta) -> bool:
        return True



