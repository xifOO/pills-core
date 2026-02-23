from pills_core._enums import ColumnRole, TransformPhase
from pills_core.strategies.numeric.imputation import (
    MeanImputation,
    MedianImputation,
    ModeImputation,
    NumericalImputationStrategy,
    UpperBoundaryImputation,
    ZeroImputation,
)
from pills_core.strategies.numeric.outliers import (
    NumericalOutlierStrategy,
    IQRStrategy,
    WinsorizeStrategy,
    ZScoreStrategy,
)
from pills_core.strategies.numeric.scaling import LogTransformStrategy, MinMaxScalerStrategy, NumericalScalingStrategy, RobustScalerStrategy, StandardScalerStrategy
from pills_core.strategies.registry import StrategyRegistry


def build_imputation_registry() -> StrategyRegistry[NumericalImputationStrategy]:
    return (
        StrategyRegistry[NumericalImputationStrategy](ColumnRole.NUMERICAL, TransformPhase.IMPUTATION)
        .register(MedianImputation())
        .register(MeanImputation())
        .register(ModeImputation())
        .register(ZeroImputation())
        .register(UpperBoundaryImputation())
    )


def build_outliers_registry() -> StrategyRegistry[NumericalOutlierStrategy]:
    return (
        StrategyRegistry[NumericalOutlierStrategy](ColumnRole.NUMERICAL, TransformPhase.OUTLIER)
        .register(IQRStrategy())
        .register(WinsorizeStrategy())
        .register(ZScoreStrategy())
    )


def build_scaling_registry() -> StrategyRegistry[NumericalScalingStrategy]:
    return (
        StrategyRegistry[NumericalScalingStrategy](ColumnRole.NUMERICAL, TransformPhase.SCALING)
        .register(StandardScalerStrategy())
        .register(MinMaxScalerStrategy())
        .register(LogTransformStrategy())
        .register(RobustScalerStrategy())
    )
