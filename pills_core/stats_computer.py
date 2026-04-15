from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional

import numpy as np
import pandas as pd

from pills_core.types.profiles import ColumnTypeProfile
from pills_core.types.stats import CategoricalColumnStats, NumericalColumnStats, StatsT


class StatsComputer(ABC, Generic[StatsT]):
    def __init__(self, sample_size: Optional[int] = None) -> None:
        self.sample_size = sample_size

    def _maybe_sample(self, series: pd.Series) -> pd.Series:
        if self.sample_size and len(series) > self.sample_size:
            return series.sample(n=self.sample_size, random_state=42)
        return series

    @abstractmethod
    def compute(self, series: pd.Series) -> StatsT: ...


class WelfordAccumulator:
    """
    Welford's online algorithm for numerically stable mean and variance.
    Avoids loading entire dataset into memory for a single pass.
    """

    __slots__ = ("_count", "_mean", "_m2")

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, chunk: pd.Series) -> None:
        for value in chunk:
            self._count += 1
            delta = value - self._mean
            self._mean += delta / self._count
            self._m2 += delta * (value - self._mean)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._m2 / (self._count - 1) if self._count > 1 else 0.0

    @property
    def std(self) -> float:
        return self.variance**0.5


class NumericalStatsComputer(StatsComputer[NumericalColumnStats]):
    def __init__(
        self, sample_size: int | None = None, chunk_size: int | None = None
    ) -> None:
        super().__init__(sample_size)
        self.chunk_size = chunk_size

    def compute(self, series: pd.Series) -> NumericalColumnStats:
        series = self._maybe_sample(series)
        clean = series.dropna()

        mean, variance, std = self._compute_mean_var(clean)

        quantiles = clean.quantile([0.05, 0.25, 0.75, 0.95])
        q1, q3 = quantiles[0.25], quantiles[0.75]
        iqr = q3 - q1

        outlier_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)

        count = len(clean)
        n_unique = clean.nunique()

        diffs = clean.sort_index().diff().dropna()

        monotonic_ratio = (
            float((diffs > 0).sum() / len(diffs)) if len(diffs) > 0 else 0.0
        )

        mode_vals = clean.mode()
        mode = mode_vals.iloc[0] if not mode_vals.empty else clean.median()

        return NumericalColumnStats(
            max=clean.max(),
            min=clean.min(),
            mean=mean,
            median=clean.median(),
            mode=mode,
            std=std,
            count=count,
            variance=variance,
            skewness=pd.to_numeric(clean.skew()),
            kurtosis=pd.to_numeric(clean.kurt()),
            range=clean.max() - clean.min(),
            n_unique=n_unique,
            missing_ratio=float(series.isna().mean()),
            outlier_ratio=float(outlier_mask.mean()),
            q1=q1,
            q3=q3,
            p05=quantiles[0.05],
            p95=quantiles[0.95],
            is_integer_valued=bool((clean == clean.round()).all()),
            monotonic_ratio=monotonic_ratio,
            cv=float(abs(std / mean)) if mean != 0 else 0.0,
            unique_ratio=float(n_unique / count) if count > 0 else 0.0,
            zero_ratio=float((clean == 0).mean()),
        )

    def _compute_mean_var(self, clean: pd.Series) -> tuple[float, float, float]:
        if self.chunk_size:
            acc = WelfordAccumulator()
            for i in range(0, len(clean), self.chunk_size):
                acc.update(clean.iloc[i : i + self.chunk_size])
            return acc.mean, acc.variance, acc.std

        mean = float(clean.mean())
        variance = float(clean.var())
        std = float(clean.std())
        return mean, variance, std


class CategoricalStatsComputer(StatsComputer[CategoricalColumnStats]):
    def __init__(
        self,
        rare_thresholds: float,
        sample_size: int | None = None,
    ) -> None:
        super().__init__(sample_size)
        self.rare_thresholds = rare_thresholds

    def compute(self, series: pd.Series) -> CategoricalColumnStats:
        series = self._maybe_sample(series)
        clean = series.dropna()

        count = len(clean)

        if count == 0:
            return CategoricalColumnStats(
                n_unique=0,
                missing_ratio=1.0,
                most_frequent="",
                most_frequent_ratio=0.0,
                rare_categories=[],
                rare_ratio=0.0,
                entropy=0.0,
                mode="",
            )

        value_counts = clean.value_counts()
        probs = value_counts / count

        n_unique = int(value_counts.shape[0])

        most_frequent = value_counts.index[0]
        most_frequent_ratio = float(probs.iloc[0])

        rare_mask = probs < self.rare_thresholds
        rare_categories = value_counts.index[rare_mask].tolist()
        rare_ratio = float(probs[rare_mask].sum())

        entropy = float(-(probs * np.log2(probs)).sum())

        mode_vals = clean.mode()
        mode = mode_vals.iloc[0] if not mode_vals.empty else most_frequent

        return CategoricalColumnStats(
            n_unique=n_unique,
            missing_ratio=float(series.isna().mean()),
            most_frequent=str(most_frequent),
            most_frequent_ratio=most_frequent_ratio,
            rare_categories=[str(x) for x in rare_categories],
            rare_ratio=rare_ratio,
            entropy=entropy,
            mode=str(mode),
        )


class StatsComputerRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, Callable[[], StatsComputer]] = {}

    def register(
        self, inferred_type: str, factory: Callable[[], StatsComputer]
    ) -> None:
        self._registry[inferred_type] = factory

    def get_computer(self, type_profile: ColumnTypeProfile) -> StatsComputer:
        if type_profile.inferred_type not in self._registry:
            raise ValueError(
                f"No StatsComputer registered for type: {type_profile.inferred_type}"
            )
        return self._registry[type_profile.inferred_type]()
