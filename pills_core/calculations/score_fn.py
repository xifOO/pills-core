from pills_core.strategies.priorities import Priority
from pills_core.types.stats import NumericalColumnStats


def median_score(stats: NumericalColumnStats) -> int:
    if abs(stats.skewness) >= 1.0 or stats.outlier_ratio > 0.05:
        return Priority.IMPUTATION_HIGH
    return Priority.IMPUTATION_LOW


def mean_score(stats: NumericalColumnStats) -> int:
    if abs(stats.skewness) < 1.0 and stats.outlier_ratio < 0.05:
        return Priority.IMPUTATION_HIGH
    return Priority.IMPUTATION_LOW


def mode_score(stats: NumericalColumnStats) -> int:
    if stats.n_unique <= 10:
        return Priority.IMPUTATION_HIGH
    return Priority.IMPUTATION_LOW


def zero_score(stats: NumericalColumnStats) -> int:
    if stats.missing_ratio > 0.3 and stats.min >= 0:
        return Priority.IMPUTATION_HIGH
    return Priority.IMPUTATION_LOW


def upper_boundary_score(stats: NumericalColumnStats) -> int:
    if stats.skewness > 2.0 and stats.outlier_ratio > 0.05:
        return Priority.IMPUTATION_HIGH
    return Priority.IMPUTATION_LOW


def lower_boundary_score(stats: NumericalColumnStats) -> int:
    if stats.skewness < -2.0 and stats.outlier_ratio > 0.05:
        return Priority.IMPUTATION_HIGH
    return Priority.IMPUTATION_LOW
