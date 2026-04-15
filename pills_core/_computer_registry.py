from pills_core.config import ComputeConfig
from pills_core.stats_computer import (
    CategoricalStatsComputer,
    NumericalStatsComputer,
    StatsComputerRegistry,
)


def build_computer_registry(config: ComputeConfig):
    registry = StatsComputerRegistry()

    registry.register(
        "numeric",
        lambda: NumericalStatsComputer(
            sample_size=config.sample_size, chunk_size=config.chunk_size
        ),
    )

    registry.register(
        "categorical",
        lambda: CategoricalStatsComputer(
            rare_thresholds=config.rare_threshold, sample_size=config.sample_size
        ),
    )

    return registry
