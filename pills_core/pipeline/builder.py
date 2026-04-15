from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from pills_core._enums import TransformPhase
from pills_core.pipeline.context import ColumnContext
from pills_core.pipeline.sequence import TransformSequence
from pills_core.pipeline.step import Step
from pills_core.stats_computer import StatsComputer
from pills_core.strategies.base import SingleStrategy
from pills_core.strategies.registry import StrategyRegistry
from pills_core.strategies.resolver import resolve_phase_order


class PipelineBuilder:
    """
    Builds a frozen TransformSequence for a single column.
    """

    def __init__(
        self,
        phase_registries: Dict[TransformPhase, StrategyRegistry],
    ) -> None:
        self._phase_registries = phase_registries

    def build(
        self, series: pd.Series, context: ColumnContext, computer: StatsComputer
    ) -> TransformSequence:
        ordered = self._resolve(context)
        return self._fit_steps(series, ordered, computer)

    def _resolve(
        self,
        context: ColumnContext,
    ) -> List[Tuple[TransformPhase, SingleStrategy]]:
        selected: Dict[TransformPhase, SingleStrategy] = {}

        for phase, registry in self._phase_registries.items():
            candidates = registry.resolve(
                context.meta, context.embedding, context.stats
            )
            if candidates:
                selected[phase] = candidates[0][0]

        phase_order = resolve_phase_order(*selected.values())
        return [(phase, selected[phase]) for phase in phase_order]

    def _fit_steps(
        self,
        series: pd.Series,
        ordered: List[Tuple[TransformPhase, SingleStrategy]],
        computer: StatsComputer,
    ) -> TransformSequence:
        steps: List[Step] = []
        current = series.copy()

        for phase, strategy in ordered:
            stats = computer.compute(current)
            step = Step(phase=phase, strategy=strategy, stats=stats)
            steps.append(step)
            current = strategy.apply(current, stats)

        return TransformSequence(steps=tuple(steps))
