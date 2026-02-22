from pills_core._enums import ColumnRole
from pills_core.strategies.numeric.imputation import (
    LOWER_BOUNDARY_IMPUTATION,
    MEAN_IMPUTATION,
    MEDIAN_IMPUTATION,
    UPPER_BOUNDARY_IMPUTATION,
    ZERO_IMPUTATION,
    ImputationStrategy,
)
from pills_core.strategies.numeric.outliers import (
    IQRStrategy,
    OutlierStrategy,
    WinsorizeStrategy,
    ZScoreStrategy,
)
from pills_core.strategies.registry import StrategyRegistry
from pills_core.types.stats import NumericalColumnStats


def build_imputation_registry() -> StrategyRegistry[ImputationStrategy]:
    return (
        StrategyRegistry[ImputationStrategy](ColumnRole.NUMERICAL)
        .register(MEDIAN_IMPUTATION)
        .register(MEAN_IMPUTATION)
        .register(ZERO_IMPUTATION)
        .register(UPPER_BOUNDARY_IMPUTATION)
        .register(LOWER_BOUNDARY_IMPUTATION)
    )


def build_outliers_registry() -> StrategyRegistry[OutlierStrategy]:
    return (
        StrategyRegistry[OutlierStrategy](ColumnRole.NUMERICAL)
        .register(IQRStrategy())
        .register(WinsorizeStrategy())
        .register(ZScoreStrategy(3.0))  # later to config
    )


def build_scaling_registry() -> StrategyRegistry: ...
