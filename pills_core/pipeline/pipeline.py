from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from pills_core._infer_types import TypeInferencer
from pills_core.analyzers import AnalyzerRegistry
from pills_core.pipeline.builder import PipelineBuilder
from pills_core.pipeline.context import ColumnContext
from pills_core.pipeline.profiler import ColumnProfiler
from pills_core.pipeline.sequence import TransformSequence
from pills_core.stats_computer import StatsComputerRegistry


@dataclass(frozen=True, slots=True)
class FittedColumnArtifact:
    context: ColumnContext
    sequence: TransformSequence


class Pipeline:
    def __init__(
        self,
        profiler: ColumnProfiler,
        builder: PipelineBuilder,
        type_inferencer: TypeInferencer,
        computer_registry: StatsComputerRegistry,
        analyzer_registry: AnalyzerRegistry,
    ) -> None:
        self._profiler = profiler
        self._builder = builder
        self._type_inferencer = type_inferencer
        self._computer_registry = computer_registry
        self._analyzer_registry = analyzer_registry

    def fit(self, series: pd.Series, is_target: bool) -> FittedColumnArtifact:
        type_profile = self._type_inferencer.infer(series)

        computer = self._computer_registry.get_computer(type_profile)
        analyzer = self._analyzer_registry.get_analyzer(type_profile)

        context = self._profiler.profile(series, is_target, analyzer, computer)
        sequence = self._builder.build(series, context, computer)  # later fix

        return FittedColumnArtifact(context=context, sequence=sequence)

    def transform(
        self,
        series: pd.Series,
        artifact: FittedColumnArtifact,
    ) -> pd.Series:
        return artifact.sequence.apply(series)

    def fit_transform(
        self, series: pd.Series, is_target: bool
    ) -> Tuple[FittedColumnArtifact, pd.Series]:
        artifact = self.fit(series, is_target)
        result = self.transform(series, artifact)
        return artifact, result
