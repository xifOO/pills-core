import pandas as pd
from pills_core._enums import ColumnRole
from pills_core.descriptors.column.column import ColumnDescriptor


class NumericDescriptor(ColumnDescriptor):
    priority = 10

    def __init__(self, column_name: str, role: ColumnRole, is_target: bool) -> None:
        super().__init__(column_name, role, is_target)

    def match(self, data: pd.Series) -> bool:
        """
        Determine whether this descriptor can handle the column.
        Should return True for numeric non-boolean dtypes.
        """
        ...

    def fit(self, data: pd.Series) -> None:
        """
        Analyze column distribution and choose transformation strategies.
        Stores computed statistics inside the descriptor.
        """
        ...

    def transform(self, data: pd.Series) -> pd.Series:
        """Apply learned transformations"""
        ...

    def explain(self, data: pd.Series) -> str: ...
