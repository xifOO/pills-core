from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from pills_core._enums import TransformPhase
from pills_core.pipeline.context import ColumnContext
from pills_core.pipeline.sequence import TransformSequence
from pills_core.pipeline.step import Step
from pills_core.pipeline.trace import PhaseTrace
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
    ) -> Tuple[TransformSequence, Tuple[PhaseTrace, ...]]:
        ordered, traces = self._resolve(context)
        sequence = self._fit_steps(series, ordered, computer)
        return sequence, tuple(traces)

    def _resolve(
        self,
        context: ColumnContext,
    ) -> Tuple[List[Tuple[TransformPhase, SingleStrategy]], List[PhaseTrace]]:
        selected: Dict[TransformPhase, SingleStrategy] = {}
        traces: List[PhaseTrace] = []

        for phase, registry in self._phase_registries.items():
            candidates = registry.resolve(
                context.meta, context.embedding, context.stats
            )
            if candidates:
                winner = candidates[0][0]
                selected[phase] = winner

                traces.append(
                    PhaseTrace(phase=phase, candidates=tuple(candidates), winner=winner)
                )

        phase_order = resolve_phase_order(*selected.values())
        ordered = [(phase, selected[phase]) for phase in phase_order]
        return ordered, traces

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
