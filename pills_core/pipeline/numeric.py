import pandas as pd

from pills_core._enums import TransformPhase
from pills_core.analyzers import NumericalColumnAnalyzer
from pills_core.calculations.numeric import compute_stats
from pills_core.pipeline.base import BasePipeline, FittedColumnArtifact
from pills_core.strategies.base import SingleStrategy
from pills_core.strategies.numeric.base import NumericalColumnMeta, NumericalEmbedding
from pills_core.strategies.numeric.imputation import NumericalImputationStrategy
from pills_core.strategies.numeric.outliers import NumericalOutlierStrategy
from pills_core.strategies.numeric.scaling import NumericalScalingStrategy
from pills_core.strategies.resolver import resolve_phase_order
from pills_core.types.stats import NumericalColumnStats


class FittedNumericalArtifact(
    FittedColumnArtifact[NumericalColumnStats, NumericalColumnMeta, NumericalEmbedding]
): ...


class NumericalColumnPipeline(
    BasePipeline[NumericalColumnStats, NumericalColumnMeta, NumericalEmbedding]
):
    def __init__(
        self,
        analyzer: NumericalColumnAnalyzer,
        imputation_strategy: NumericalImputationStrategy,
        outlier_strategy: NumericalOutlierStrategy,
        scaling_stratagy: NumericalScalingStrategy,
    ) -> None:
        self.analyzer = analyzer
        self.imputation_strategy = imputation_strategy
        self.outlier_strategy = outlier_strategy
        self.scaling_stratagy = scaling_stratagy

    def fit(self, series: pd.Series, is_target: bool) -> FittedNumericalArtifact:
        stats = compute_stats(series)
        meta = self.analyzer.build_meta(series, stats, is_target)
        embedding = self.analyzer.build_column_embedding(stats, meta)

        phase_order = resolve_phase_order(self.imputation_strategy, self.outlier_strategy, self.scaling_stratagy)

        strategies = {
            TransformPhase.IMPUTATION: self.imputation_strategy,
            TransformPhase.OUTLIER: self.outlier_strategy,
            TransformPhase.SCALING: self.scaling_stratagy,
        }

        current = series.copy()
        phase_stats = {}

        for phase in phase_order:
            phase_stats[phase] = compute_stats(current)
            strategy: SingleStrategy = strategies[phase]
            current = strategy.apply(current, phase_stats[phase])

        return FittedNumericalArtifact(
            name=str(series.name),
            meta=meta,
            embedding=embedding,
            phase_order=tuple(phase_order),
            strategies=strategies,
            phase_stats=phase_stats,
        )

    def transform(self, series: pd.Series, artifact: FittedColumnArtifact) -> pd.Series:
        result = series.copy()
        for phase in artifact.phase_order:
            result = artifact.strategies[phase].apply(
                result, artifact.phase_stats[phase]
            )
        return result
