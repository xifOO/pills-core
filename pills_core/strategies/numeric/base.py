from abc import ABC

from pills_core._enums import ColumnRole
from pills_core.strategies.base import SingleStrategy
from pills_core.types.stats import NumericalColumnStats


class NumericalStrategy(SingleStrategy[NumericalColumnStats], ABC):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    @property
    def column_type(self) -> ColumnRole:
        return ColumnRole.NUMERICAL
