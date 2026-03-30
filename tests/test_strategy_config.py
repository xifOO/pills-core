from pills_core.config import PillConfig
from pills_core.strategies.config import (
    BoxCoxConfig,
    IQRStrategyConfig,
    LogTransformConfig,
    LowerBoundaryImputationConfig,
    MeanImputationConfig,
    MedianImputationConfig,
    MinMaxScalerConfig,
    ModeImputationConfig,
    NumericalImputationRegistryConfig,
    NumericalImputationStrategiesConfig,
    NumericalOutlierRegistryConfig,
    NumericalOutlierStrategiesConfig,
    NumericalScalingRegistryConfig,
    NumericalScalingStrategiesConfig,
    NumericStrategiesConfig,
    RobustScalerConfig,
    SqrtTransformConfig,
    StandardScalerConfig,
    StrategiesConfig,
    StrategyWeightsConfig,
    UpperBoundaryImputationConfig,
    WinsorizeStrategyConfig,
    ZeroImputationConfig,
    ZScoreStrategyConfig,
)
from pills_core.strategies.numeric._registry import (
    build_imputation_registry,
    build_outliers_registry,
    build_scaling_registry,
)
from pills_core.strategies.numeric.imputation import MeanImputation
from pills_core.strategies.numeric.outliers import ZScoreStrategy
from pills_core.strategies.numeric.scaling import BoxCoxStrategy


def test_build_imputation_registry_respects_custom_config():
    registry = build_imputation_registry(
        NumericalImputationRegistryConfig(
            weights=StrategyWeightsConfig(missing_ratio_fit=3.0),
            strategies=NumericalImputationStrategiesConfig(
                median=MedianImputationConfig(enabled=False),
                mean=MeanImputationConfig(
                    radius=0.9,
                    max_outlier_ratio=0.25,
                    max_abs_skewness=2.5,
                ),
                mode=ModeImputationConfig(enabled=False),
                constant_zero=ZeroImputationConfig(enabled=False),
                upper_boundary=UpperBoundaryImputationConfig(enabled=False),
                lower_boundary=LowerBoundaryImputationConfig(enabled=False),
            ),
        )
    )

    assert registry.weights["missing_ratio_fit"] == 3.0
    assert len(registry.strategies) == 1

    strategy = registry.strategies[0]
    assert isinstance(strategy, MeanImputation)
    assert strategy.radius == 0.9
    assert strategy.embedding.outliers_sensitivity == 0.9
    assert strategy.sensitive_outlier_ratio_limit == 0.25
    assert strategy.sensitive_skewness_limit == 2.5


def test_build_outliers_registry_applies_runtime_parameters():
    registry = build_outliers_registry(
        NumericalOutlierRegistryConfig(
            strategies=NumericalOutlierStrategiesConfig(
                iqr=IQRStrategyConfig(enabled=False),
                winsorize=WinsorizeStrategyConfig(enabled=False),
                z_score=ZScoreStrategyConfig(
                    radius=0.7,
                    threshold=4.5,
                    max_abs_skewness=1.7,
                    min_sample_size=42,
                ),
            )
        )
    )

    assert len(registry.strategies) == 1

    strategy = registry.strategies[0]
    assert isinstance(strategy, ZScoreStrategy)
    assert strategy.radius == 0.7
    assert strategy.embedding.distribution_preservation == 0.7
    assert strategy.threshold == 4.5
    assert strategy.normality_skewness_limit == 1.7
    assert strategy.min_sample_size == 42


def test_build_scaling_registry_accepts_nested_config_from_pill_config():
    app_config = PillConfig(
        strategies=StrategiesConfig(
            numeric=NumericStrategiesConfig(
                scaling=NumericalScalingRegistryConfig(
                    strategies=NumericalScalingStrategiesConfig(
                        standard_scaler=StandardScalerConfig(enabled=False),
                        min_max_scaler=MinMaxScalerConfig(enabled=False),
                        log_transform=LogTransformConfig(enabled=False),
                        robust_scaler=RobustScalerConfig(enabled=False),
                        box_cox=BoxCoxConfig(
                            radius=0.8,
                            min_skewness=3.5,
                            shift_epsilon=1e-4,
                        ),
                        sqrt_transform=SqrtTransformConfig(enabled=False),
                    )
                )
            )
        )
    )

    registry = build_scaling_registry(app_config.strategies.numeric.scaling)

    assert len(registry.strategies) == 1

    strategy = registry.strategies[0]
    assert isinstance(strategy, BoxCoxStrategy)
    assert strategy.radius == 0.8
    assert strategy.embedding.outliers_sensitivity == 0.7
    assert strategy.min_skewness == 3.5
    assert strategy.shift_epsilon == 1e-4
