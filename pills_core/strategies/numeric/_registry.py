from typing import Optional, Type

from pills_core._enums import ColumnRole, TransformPhase
from pills_core.strategies.base import SingleStrategy
from pills_core.strategies.config import (
    NumericalImputationRegistryConfig,
    NumericalOutlierRegistryConfig,
    NumericalScalingRegistryConfig,
    NumericalStrategyInstanceConfig,
)
from pills_core.strategies.numeric.imputation import (
    LowerBoundaryImputation,
    MeanImputation,
    MedianImputation,
    ModeImputation,
    UpperBoundaryImputation,
    ZeroImputation,
)
from pills_core.strategies.numeric.outliers import (
    IQRStrategy,
    WinsorizeStrategy,
    ZScoreStrategy,
)
from pills_core.strategies.numeric.scaling import (
    BoxCoxStrategy,
    LogTransformStrategy,
    MinMaxScalerStrategy,
    RobustScalerStrategy,
    SqrtTransformStrategy,
    StandardScalerStrategy,
)
from pills_core.strategies.registry import StrategyRegistry


def _build_if_enabled(
    strategy: Type[SingleStrategy],
    cfg: NumericalStrategyInstanceConfig,
) -> Optional[SingleStrategy]:
    if not cfg.enabled:
        return None

    try:
        return strategy(
            embedding=cfg.embedding.as_embedding(),
            **cfg.model_dump(by_alias=True, exclude={"enabled", "embedding"}),
        )

    except TypeError as e:
        raise ValueError(f"{strategy.__name__} config mismatch: {e}")


def build_imputation_registry(
    config: NumericalImputationRegistryConfig,
) -> StrategyRegistry:
    s = config.strategies

    strategies = [
        _build_if_enabled(
            MedianImputation,
            s.median,
        ),
        _build_if_enabled(
            MeanImputation,
            s.mean,
        ),
        _build_if_enabled(
            ModeImputation,
            s.mode,
        ),
        _build_if_enabled(
            ZeroImputation,
            s.constant_zero,
        ),
        _build_if_enabled(
            UpperBoundaryImputation,
            s.upper_boundary,
        ),
        _build_if_enabled(
            LowerBoundaryImputation,
            s.lower_boundary,
        ),
    ]

    return StrategyRegistry(
        ColumnRole.NUMERICAL,
        TransformPhase.IMPUTATION,
        config.weights.as_dict(),
    ).bulk_register(strategies)


def build_outliers_registry(
    config: NumericalOutlierRegistryConfig,
) -> StrategyRegistry:
    s = config.strategies

    strategies = [
        _build_if_enabled(
            IQRStrategy,
            s.iqr,
        ),
        _build_if_enabled(
            WinsorizeStrategy,
            s.winsorize,
        ),
        _build_if_enabled(
            ZScoreStrategy,
            s.z_score,
        ),
    ]

    return StrategyRegistry(
        ColumnRole.NUMERICAL,
        TransformPhase.OUTLIER,
        config.weights.as_dict(),
    ).bulk_register(strategies)


def build_scaling_registry(
    config: NumericalScalingRegistryConfig,
) -> StrategyRegistry:
    s = config.strategies

    strategies = [
        _build_if_enabled(
            StandardScalerStrategy,
            s.standard_scaler,
        ),
        _build_if_enabled(
            MinMaxScalerStrategy,
            s.min_max_scaler,
        ),
        _build_if_enabled(
            LogTransformStrategy,
            s.log_transform,
        ),
        _build_if_enabled(
            RobustScalerStrategy,
            s.robust_scaler,
        ),
        _build_if_enabled(
            BoxCoxStrategy,
            s.box_cox,
        ),
        _build_if_enabled(
            SqrtTransformStrategy,
            s.sqrt_transform,
        ),
    ]

    return StrategyRegistry(
        ColumnRole.NUMERICAL,
        TransformPhase.SCALING,
        config.weights.as_dict(),
    ).bulk_register(strategies)
