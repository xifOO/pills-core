from typing import Final

from pills_core._enums import ColumnRole, TransformPhase
from pills_core.strategies.numeric.imputation import (
    LowerBoundaryImputation,
    MeanImputation,
    MedianImputation,
    ModeImputation,
    NumericalImputationStrategy,
    UpperBoundaryImputation,
    ZeroImputation,
)
from pills_core.strategies.numeric.outliers import (
    IQRStrategy,
    NumericalOutlierStrategy,
    WinsorizeStrategy,
    ZScoreStrategy,
)
from pills_core.strategies.numeric.scaling import (
    BoxCoxStrategy,
    LogTransformStrategy,
    MinMaxScalerStrategy,
    NumericalScalingStrategy,
    RobustScalerStrategy,
    SqrtTransformStrategy,
    StandardScalerStrategy,
)
from pills_core.strategies.registry import StrategyRegistry

IMPUTATION_WEIGHTS: Final = {
    "skewness_sensitivity": 0.8,
    "outliers_sensitivity": 0.4,
    "missing_ratio_fit": 1.5,
    "distribution_preservation": 0.8,
    "target_safety": 0.0,
    "cardinality_fit": 0.5,
}

OUTLIER_WEIGHTS: Final = {
    "skewness_sensitivity": 1.2,
    "outliers_sensitivity": 2.0,
    "missing_ratio_fit": 0.1,
    "distribution_preservation": 0.8,
    "target_safety": 0.0,
    "cardinality_fit": 0.3,
}

SCALING_WEIGHTS: Final = {
    "skewness_sensitivity": 1.5,
    "outliers_sensitivity": 1.8,
    "missing_ratio_fit": 0.1,
    "distribution_preservation": 0.6,
    "target_safety": 0.5,
    "cardinality_fit": 0.3,
}


def build_imputation_registry() -> StrategyRegistry[NumericalImputationStrategy]:
    return StrategyRegistry[NumericalImputationStrategy](
        ColumnRole.NUMERICAL, TransformPhase.IMPUTATION, IMPUTATION_WEIGHTS
    ).bulk_register(
        [
            MedianImputation(),
            MeanImputation(),
            ModeImputation(),
            ZeroImputation(),
            UpperBoundaryImputation(),
            LowerBoundaryImputation(),
        ]
    )


def build_outliers_registry() -> StrategyRegistry[NumericalOutlierStrategy]:
    return StrategyRegistry[NumericalOutlierStrategy](
        ColumnRole.NUMERICAL, TransformPhase.OUTLIER, OUTLIER_WEIGHTS
    ).bulk_register([IQRStrategy(), WinsorizeStrategy(), ZScoreStrategy()])


def build_scaling_registry() -> StrategyRegistry[NumericalScalingStrategy]:
    return StrategyRegistry[NumericalScalingStrategy](
        ColumnRole.NUMERICAL, TransformPhase.SCALING, SCALING_WEIGHTS
    ).bulk_register(
        [
            StandardScalerStrategy(),
            MinMaxScalerStrategy(),
            LogTransformStrategy(),
            RobustScalerStrategy(),
            BoxCoxStrategy(),
            SqrtTransformStrategy(),
        ]
    )
