from __future__ import annotations

from typing import Dict

import pandas as pd

from pills_core._enums import TransformPhase
from pills_core.analyzers import NumericalColumnAnalyzer
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
        phase_registries: Dict[TransformPhase, StrategyRegistry],
    ) -> None:
        self._analyzer = analyzer
        self._phase_registries = phase_registries

    def compute_stats(self, series: pd.Series) -> NumericalColumnStats:
        from pills_core.calculations.numeric import (
            compute_stats,  # late import - fix generic
        )

        return compute_stats(series)

    def fit(self, series: pd.Series, is_target: bool) -> FittedNumericalArtifact:
        stats = self.compute_stats(series)
        meta = self._analyzer.build_meta(series, stats, is_target)
        embedding = self._analyzer.build_column_embedding(stats, meta)

        selected = self._resolve_strategies(meta, embedding, stats)

        current = series.copy()
        phase_order = resolve_phase_order(*selected.values())
        strategies: Dict[TransformPhase, SingleStrategy] = {}
        phase_stats: Dict[TransformPhase, NumericalColumnStats] = {}

        for phase in phase_order:
            strategy = selected[phase]
            phase_stats[phase] = self.compute_stats(current)
            current = strategy.apply(current, phase_stats[phase])

            strategies[phase] = strategy

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

    def _resolve_strategies(
        self,
        meta: NumericalColumnMeta,
        embedding: NumericalEmbedding,
        stats: NumericalColumnStats,
    ) -> Dict[TransformPhase, SingleStrategy]:
        selected = {}

        for phase, registry in self._phase_registries.items():
            ordered = registry.resolve(meta, embedding, stats)
            selected[phase] = ordered[0][0]

        return selected
