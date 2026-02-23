from dataclasses import dataclass
from pills_core.types.stats import NumericalColumnStats


@dataclass(frozen=True)
class Score:
    """
    Priority assembled from components.
    base      — how good the strategy is in general
    condition — how well the current stats fit this strategy
    penalty   — risk penalty (outliers, skewness, small sample)

    Scale guide:
        base:      300 (strong general strategy)
                   250 (good but situational)
                   150 (transformation — changes shape, use carefully)
        condition: 150 (perfect match)
                   100 (strong match)
                    75 (moderate match)
                    25 (minor bonus)
        penalty:   150 (critical mismatch — should rarely win)
                   100 (significant risk)
                    50 (minor concern)
    """
    base: int
    condition: int
    penalty: int = 0

    @property
    def total(self) -> int:
        return self.base + self.condition - self.penalty

    def __int__(self) -> int:
        return self.total

    def __repr__(self) -> str:
        return (
            f"Score(base={self.base}, condition={self.condition}, "
            f"penalty={self.penalty} → total={self.total})"
        )


def for_median(stats: NumericalColumnStats) -> Score:
    condition = 0
    if abs(stats.skewness) >= 1.0:
        condition += 100  # median is robust to skewness — strong match
    if stats.outlier_ratio > 0.05:
        condition += 75   # median is robust to outliers
    return Score(base=300, condition=condition)


def for_mean(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if abs(stats.skewness) < 0.5:
        condition += 100  # mean == median when symmetric
    if stats.outlier_ratio > 0.05:
        penalty += 150    # mean is pulled hard by outliers
    if abs(stats.skewness) >= 1.0:
        penalty += 100    # mean != center when skewed
    return Score(base=300, condition=condition, penalty=penalty)


def for_mode(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if stats.n_unique <= 10:
        condition += 100  # mode makes sense for low-cardinality
    if stats.n_unique > 20:
        penalty += 150    # mode is meaningless for high-cardinality numeric
    return Score(base=200, condition=condition, penalty=penalty)


def for_zero(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if stats.missing_ratio > 0.3:
        condition += 75   # large gaps — zero is safe fallback
    if stats.min >= 0:
        condition += 25   # zero is in-domain for non-negative
    if stats.mean > 0 and stats.missing_ratio < 0.1:
        penalty += 100    # distorts distribution when data is mostly present
    return Score(base=150, condition=condition, penalty=penalty)


def for_upper_boundary(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if stats.skewness > 2.0:
        condition += 100  # right tail — filling with upper boundary makes sense
    if stats.outlier_ratio > 0.05:
        condition += 75
    if stats.outlier_ratio > 0.15:
        penalty += 75     # boundary itself is distorted by too many outliers
    return Score(base=150, condition=condition, penalty=penalty)


def for_lower_boundary(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if stats.skewness < -2.0:
        condition += 100  # left tail — filling with lower boundary makes sense
    if stats.outlier_ratio > 0.05:
        condition += 75
    if stats.outlier_ratio > 0.15:
        penalty += 75
    return Score(base=150, condition=condition, penalty=penalty)


def for_iqr(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if abs(stats.skewness) < 1.0:
        condition += 100  # IQR is symmetric — best for symmetric distributions
    if stats.outlier_ratio < 0.05:
        condition += 75   # few outliers — gentle clip is enough
    if abs(stats.skewness) >= 1.5:
        penalty += 100    # IQR fences become wrong for skewed distributions
    return Score(base=250, condition=condition, penalty=penalty)


def for_winsorize(stats: NumericalColumnStats) -> Score:
    condition = 0
    if abs(stats.skewness) >= 1.0:
        condition += 100  # p05/p95 handles skewed better than IQR fences
    if stats.outlier_ratio >= 0.05:
        condition += 100  # many outliers — winsorize clips both tails by ratio
    if stats.count < 20:
        condition -= 75   # percentiles unstable on small samples
    return Score(base=250, condition=condition)


def for_zscore(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if abs(stats.skewness) < 0.5:
        condition += 125  # z-score most meaningful when distribution is normal
    if stats.count >= 100:
        condition += 50   # std is stable on large samples
    if abs(stats.skewness) >= 1.0:
        penalty += 125    # std is distorted when skewed
    if stats.count < 30:
        penalty += 100    # std unstable on small samples
    return Score(base=250, condition=condition, penalty=penalty)


def for_standard_scaler(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if abs(stats.skewness) < 0.5:
        condition += 100  # best when symmetric — mean is accurate center
    if stats.outlier_ratio < 0.02:
        condition += 75   # clean data — mean/std are reliable
    if stats.outlier_ratio >= 0.05:
        penalty += 150    # outliers destroy mean and std completely
    if abs(stats.skewness) >= 1.0:
        penalty += 100    # mean != center when skewed
    return Score(base=250, condition=condition, penalty=penalty)


def for_robust_scaler(stats: NumericalColumnStats) -> Score:
    condition = 0
    if stats.outlier_ratio > 0.05:
        condition += 100  # median/IQR not affected by outliers
    if abs(stats.skewness) >= 1.0:
        condition += 75   # works for any distribution shape
    # no penalties — robust is a safe fallback for any situation
    return Score(base=250, condition=condition)


def for_minmax_scaler(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if stats.outlier_ratio == 0:
        condition += 125  # min/max are accurate only without outliers
    if abs(stats.skewness) < 0.5:
        condition += 50
    if stats.outlier_ratio > 0:
        penalty += 150    # even one outlier collapses the [0,1] range
    return Score(base=200, condition=condition, penalty=penalty)


def for_log_transform(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if stats.skewness > 3.0:
        condition += 150  # heavy right tail — log is ideal
    elif stats.skewness > 1.5:
        condition += 75   # moderate right tail — log helps
    if stats.min == 0:
        condition += 25   # log1p(0) = 0, zero-safe
    if stats.outlier_ratio > 0.1:
        penalty += 50     # log compresses but doesn't remove outliers
    # base lower — log changes distribution shape, use only when needed
    return Score(base=150, condition=condition, penalty=penalty)


def for_sqrt_transform(stats: NumericalColumnStats) -> Score:
    condition = 0
    penalty = 0
    if 0.75 <= stats.skewness <= 1.5:
        condition += 125  # softer than log — exactly for moderate skewness
    if stats.min == 0:
        condition += 25   # sqrt(0) = 0, zero-safe
    if stats.skewness > 2.0:
        penalty += 75     # log handles heavy skewness better
    return Score(base=150, condition=condition, penalty=penalty)