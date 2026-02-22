from pills_core._enums import ColumnRole
from pills_core.strategies.numeric.imputation import (
    LOWER_BOUNDARY_IMPUTATION,
    MEAN_IMPUTATION,
    MEDIAN_IMPUTATION,
    UPPER_BOUNDARY_IMPUTATION,
    ZERO_IMPUTATION,
)
from pills_core.strategies.numeric.outliers import IQRStrategy, WinsorizeStrategy
from pills_core.strategies.registry import StrategyRegistry
from pills_core.types.stats import NumericalColumnStats


def build_numerical_registry() -> StrategyRegistry[NumericalColumnStats]:
    return (
        StrategyRegistry(ColumnRole.NUMERICAL)
        .register(MEDIAN_IMPUTATION)
        .register(MEAN_IMPUTATION)
        .register(ZERO_IMPUTATION)
        .register(UPPER_BOUNDARY_IMPUTATION)
        .register(LOWER_BOUNDARY_IMPUTATION)
        .register(IQRStrategy())
        .register(WinsorizeStrategy())
    )
