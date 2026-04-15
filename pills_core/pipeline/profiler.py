import pandas as pd

from pills_core.analyzers import ColumnAnalyzer
from pills_core.pipeline.context import ColumnContext
from pills_core.stats_computer import StatsComputer


class ColumnProfiler:
    def __init__(self) -> None: ...  # later: settings add

    def profile(
        self,
        series: pd.Series,
        is_target: bool,
        analyzer: ColumnAnalyzer,
        computer: StatsComputer,
    ) -> ColumnContext:
        stats = computer.compute(series)
        meta = analyzer.build_meta(series, stats, is_target)
        embedding = analyzer.build_column_embedding(stats, meta)

        return ColumnContext(
            name=str(series.name),
            stats=stats,
            meta=meta,
            embedding=embedding,
        )
