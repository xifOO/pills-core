from typing import Union

from pydantic import BaseModel, ConfigDict, Field

from pills_core.strategies.base import StrategyEmbedding
from pills_core.strategies.categorical.base import CategoricalEmbedding
from pills_core.strategies.numeric.base import NumericalEmbedding


class StrategyConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrategyWeightsConfig(StrategyConfigModel):
    skewness_sensitivity: float = 1.0
    outliers_sensitivity: float = 1.0
    missing_ratio_fit: float = 1.0
    distribution_preservation: float = 1.0
    target_safety: float = 1.0
    cardinality_fit: float = 1.0
    rare_categories_handling: float = 1.0
    imbalance_sensitivity: float = 1.0
    order_awareness: float = 1.0
    typo_tolerance: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.model_dump().items()}


class StrategyEmbeddingConfig(StrategyConfigModel):
    missing_ratio_fit: float = 1.0
    distribution_preservation: float = 1.0
    target_safety: float = 1.0
    cardinality_fit: float = 1.0


class NumericalEmbeddingConfig(StrategyEmbeddingConfig):
    skewness_sensitivity: float = 1.0
    outliers_sensitivity: float = 1.0

    def as_embedding(self) -> NumericalEmbedding:
        return NumericalEmbedding(**self.model_dump())


class CategoricalEmbeddingConfig(StrategyEmbeddingConfig):
    rare_categories_handling: float = 1.0
    imbalance_sensitivity: float = 1.0
    order_awareness: float = 1.0
    typo_tolerance: float = 1.0

    def as_embedding(self) -> CategoricalEmbedding:
        return CategoricalEmbedding(**self.model_dump())


class StrategyInstanceConfig(StrategyConfigModel):
    enabled: bool = True
    radius: float = Field(default=1.0, ge=0.0)


class NumericalStrategyInstanceConfig(StrategyInstanceConfig):
    embedding: NumericalEmbeddingConfig = Field(default_factory=NumericalEmbeddingConfig)


class SensitiveNumericalImputationConfig(NumericalStrategyInstanceConfig):
    max_outlier_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        serialization_alias="sensitive_outlier_ratio_limit",
    )
    max_abs_skewness: float = Field(
        default=1.0,
        ge=0.0,
        serialization_alias="sensitive_skewness_limit",
    )


class BoundaryImputationConfig(NumericalStrategyInstanceConfig):
    std_multiplier: float = Field(default=3.0, gt=0.0)


class MedianImputationConfig(NumericalStrategyInstanceConfig):
    radius: float = 1.9
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.9,
            outliers_sensitivity=0.1,
            missing_ratio_fit=0.8,
            distribution_preservation=0.9,
            target_safety=1.0,
            cardinality_fit=0.0,
        )
    )


class MeanImputationConfig(SensitiveNumericalImputationConfig):
    radius: float = 1.2
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.2,
            outliers_sensitivity=0.9,
            missing_ratio_fit=0.7,
            distribution_preservation=0.8,
            target_safety=1.0,
            cardinality_fit=0.5,
        )
    )


class ModeImputationConfig(NumericalStrategyInstanceConfig):
    radius: float = 1.2
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.5,
            outliers_sensitivity=0.1,
            missing_ratio_fit=0.6,
            distribution_preservation=0.3,
            target_safety=1.0,
            cardinality_fit=0.9,
        )
    )


class ZeroImputationConfig(NumericalStrategyInstanceConfig):
    radius: float = 1.0
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.6,
            outliers_sensitivity=0.4,
            missing_ratio_fit=0.9,
            distribution_preservation=0.2,
            target_safety=1.0,
            cardinality_fit=0.4,
        )
    )


class UpperBoundaryImputationConfig(BoundaryImputationConfig):
    radius: float = 1.1
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.8,
            outliers_sensitivity=0.6,
            missing_ratio_fit=0.5,
            distribution_preservation=0.2,
            target_safety=0.0,
            cardinality_fit=0.3,
        )
    )


class LowerBoundaryImputationConfig(BoundaryImputationConfig):
    radius: float = 1.1
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.8,
            outliers_sensitivity=0.6,
            missing_ratio_fit=0.5,
            distribution_preservation=0.2,
            target_safety=0.0,
            cardinality_fit=0.3,
        )
    )


class IQRStrategyConfig(NumericalStrategyInstanceConfig):
    max_abs_skewness: float = Field(default=1.5, ge=0.0)
    min_outlier_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    clip_multiplier: float = Field(default=1.5, gt=0.0)
    radius: float = 1.4
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.4,
            outliers_sensitivity=0.1,
            missing_ratio_fit=0.5,
            distribution_preservation=0.7,
            target_safety=0.0,
            cardinality_fit=0.4,
        )
    )


class ZScoreStrategyConfig(NumericalStrategyInstanceConfig):
    threshold: float = Field(default=3.0, gt=0.0)
    max_abs_skewness: float = Field(default=1.0, ge=0.0)
    min_sample_size: int = Field(default=30, ge=1)
    radius: float = 1.1
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.1,
            outliers_sensitivity=0.8,
            missing_ratio_fit=0.5,
            distribution_preservation=0.7,
            target_safety=0.0,
            cardinality_fit=0.3,
        )
    )


class WinsorizeStrategyConfig(NumericalStrategyInstanceConfig):
    min_outlier_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    min_sample_size: int = Field(default=20, ge=1)
    lower_quantile: float = Field(default=0.05, ge=0.0, lt=1.0)
    upper_quantile: float = Field(default=0.95, gt=0.0, le=1.0)
    radius: float = 1.6
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.8,
            outliers_sensitivity=0.2,
            missing_ratio_fit=0.5,
            distribution_preservation=0.6,
            target_safety=0.0,
            cardinality_fit=0.3,
        )
    )


class NormalityAwareScalingConfig(NumericalStrategyInstanceConfig):
    max_abs_skewness: float = Field(
        default=1.0,
        ge=0.0,
        serialization_alias="normality_skewness_limit",
    )


class StandardScalerConfig(NormalityAwareScalingConfig):
    radius: float = 1.2
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.1,
            outliers_sensitivity=0.9,
            missing_ratio_fit=0.5,
            distribution_preservation=1.0,
            target_safety=0.0,
            cardinality_fit=0.4,
        )
    )


class MinMaxScalerConfig(NumericalStrategyInstanceConfig):
    radius: float = 1.0
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.3,
            outliers_sensitivity=1.0,
            missing_ratio_fit=0.5,
            distribution_preservation=1.0,
            target_safety=0.0,
            cardinality_fit=0.4,
        )
    )


class LogTransformConfig(NumericalStrategyInstanceConfig):
    min_skewness: float = Field(default=1.5)
    radius: float = 1.1
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=1.0,
            outliers_sensitivity=0.3,
            missing_ratio_fit=0.5,
            distribution_preservation=0.1,
            target_safety=0.0,
            cardinality_fit=0.6,
        )
    )


class RobustScalerConfig(NumericalStrategyInstanceConfig):
    radius: float = 1.5
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.5,
            outliers_sensitivity=0.1,
            missing_ratio_fit=0.5,
            distribution_preservation=0.9,
            target_safety=0.0,
            cardinality_fit=0.4,
        )
    )


class BoxCoxConfig(NumericalStrategyInstanceConfig):
    min_skewness: float = Field(default=1.5)
    shift_epsilon: float = Field(default=1e-6, gt=0.0)
    radius: float = 1.0
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=1.0,
            outliers_sensitivity=0.7,
            missing_ratio_fit=0.5,
            distribution_preservation=0.1,
            target_safety=0.0,
            cardinality_fit=0.3,
        )
    )


class SqrtTransformConfig(NumericalStrategyInstanceConfig):
    radius: float = 1.2
    embedding: NumericalEmbeddingConfig = Field(
        default_factory=lambda: NumericalEmbeddingConfig(
            skewness_sensitivity=0.7,
            outliers_sensitivity=0.5,
            missing_ratio_fit=0.5,
            distribution_preservation=0.3,
            target_safety=0.0,
            cardinality_fit=0.5,
        )
    )


class NumericalImputationStrategiesConfig(StrategyConfigModel):
    median: MedianImputationConfig = Field(default_factory=MedianImputationConfig)
    mean: MeanImputationConfig = Field(default_factory=MeanImputationConfig)
    mode: ModeImputationConfig = Field(default_factory=ModeImputationConfig)
    constant_zero: ZeroImputationConfig = Field(default_factory=ZeroImputationConfig)
    upper_boundary: UpperBoundaryImputationConfig = Field(
        default_factory=UpperBoundaryImputationConfig
    )
    lower_boundary: LowerBoundaryImputationConfig = Field(
        default_factory=LowerBoundaryImputationConfig
    )


class NumericalImputationRegistryConfig(StrategyConfigModel):
    weights: StrategyWeightsConfig = Field(
        default_factory=lambda: StrategyWeightsConfig(
            skewness_sensitivity=0.8,
            outliers_sensitivity=0.4,
            missing_ratio_fit=1.5,
            distribution_preservation=0.8,
            target_safety=0.0,
            cardinality_fit=0.5,
        )
    )
    strategies: NumericalImputationStrategiesConfig = Field(
        default_factory=NumericalImputationStrategiesConfig
    )


class NumericalOutlierStrategiesConfig(StrategyConfigModel):
    iqr: IQRStrategyConfig = Field(default_factory=IQRStrategyConfig)
    winsorize: WinsorizeStrategyConfig = Field(default_factory=WinsorizeStrategyConfig)
    z_score: ZScoreStrategyConfig = Field(default_factory=ZScoreStrategyConfig)


class NumericalOutlierRegistryConfig(StrategyConfigModel):
    weights: StrategyWeightsConfig = Field(
        default_factory=lambda: StrategyWeightsConfig(
            skewness_sensitivity=1.2,
            outliers_sensitivity=2.0,
            missing_ratio_fit=0.1,
            distribution_preservation=0.8,
            target_safety=0.0,
            cardinality_fit=0.3,
        )
    )
    strategies: NumericalOutlierStrategiesConfig = Field(
        default_factory=NumericalOutlierStrategiesConfig
    )


class NumericalScalingStrategiesConfig(StrategyConfigModel):
    standard_scaler: StandardScalerConfig = Field(default_factory=StandardScalerConfig)
    min_max_scaler: MinMaxScalerConfig = Field(default_factory=MinMaxScalerConfig)
    log_transform: LogTransformConfig = Field(default_factory=LogTransformConfig)
    robust_scaler: RobustScalerConfig = Field(default_factory=RobustScalerConfig)
    box_cox: BoxCoxConfig = Field(default_factory=BoxCoxConfig)
    sqrt_transform: SqrtTransformConfig = Field(default_factory=SqrtTransformConfig)


class NumericalScalingRegistryConfig(StrategyConfigModel):
    weights: StrategyWeightsConfig = Field(
        default_factory=lambda: StrategyWeightsConfig(
            skewness_sensitivity=1.5,
            outliers_sensitivity=1.8,
            missing_ratio_fit=0.1,
            distribution_preservation=0.6,
            target_safety=0.5,
            cardinality_fit=0.3,
        )
    )
    strategies: NumericalScalingStrategiesConfig = Field(
        default_factory=NumericalScalingStrategiesConfig
    )


class NumericStrategiesConfig(StrategyConfigModel):
    imputation: NumericalImputationRegistryConfig = Field(
        default_factory=NumericalImputationRegistryConfig
    )
    outlier: NumericalOutlierRegistryConfig = Field(
        default_factory=NumericalOutlierRegistryConfig
    )
    scaling: NumericalScalingRegistryConfig = Field(
        default_factory=NumericalScalingRegistryConfig
    )


class StrategiesConfig(StrategyConfigModel):
    numeric: NumericStrategiesConfig = Field(default_factory=NumericStrategiesConfig)
