import pandas as pd

from pills_core.types.stats import NumericalColumnStats


def median_fill(data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
    return data.fillna(stats.median)


def mean_fill(data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
    return data.fillna(stats.mean)


def mode_fill(data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
    return data.fillna(stats.mode)


def zero_fill(data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
    return data.fillna(0)


def upper_boundary_fill(data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
    return data.fillna(stats.mean + 3 * stats.std)


def lower_boundary_fill(data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
    return data.fillna(stats.mean - 3 * stats.std)
