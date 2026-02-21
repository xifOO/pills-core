import pandas as pd
from pills_core._enums import ColumnRole
from pills_core.descriptors.base import Descriptor


class ColumnDescriptor(Descriptor[pd.Series]):
    priority: int = 0

    def __init__(
        self,
        column_name: str,
        role: ColumnRole,
        is_target: bool,
    ) -> None:
        self.column_name = column_name
        self.role = role
        self.is_target = is_target
