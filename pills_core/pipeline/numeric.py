import pandas as pd

from pills_core._enums import TransformPhase
from pills_core.analyzers import NumericalColumnAnalyzer
from pills_core.calculations.numeric import compute_stats
from pills_core.pipeline.base import BasePipeline, FittedColumnArtifact
from pills_core.strategies.base import SingleStrategy
from pills_core.strategies.numeric.base import NumericalColumnMeta, NumericalEmbedding
from pills_core.strategies.registry import StrategyRegistry
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
        imputation_registry: StrategyRegistry,
        outliers_registry: StrategyRegistry,
        scaling_registry: StrategyRegistry,
    ) -> None:
        self.analyzer = analyzer
        self.imputation_registry = imputation_registry
        self.outliers_registry = outliers_registry
        self.scaling_registry = scaling_registry

    def fit(self, series: pd.Series, is_target: bool) -> FittedNumericalArtifact:
        stats = compute_stats(series)
        meta = self.analyzer.build_meta(series, stats, is_target)
        embedding = self.analyzer.build_column_embedding(stats, meta)

        best_imputation = self.imputation_registry.resolve(meta, embedding, stats)[0][0]
        best_outlier = self.outliers_registry.resolve(meta, embedding, stats)[0][0]
        best_scaling = self.scaling_registry.resolve(meta, embedding, stats)[0][0]

        phase_order = resolve_phase_order(best_imputation, best_outlier, best_scaling)

        strategies = {
            TransformPhase.IMPUTATION: best_imputation,
            TransformPhase.OUTLIER: best_outlier,
            TransformPhase.SCALING: best_scaling,
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
