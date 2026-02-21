from abc import ABC, abstractmethod
from enum import Enum, auto
import pandas as pd
import numpy as np

from pills_core.types.stats import NumericalColumnStats


class OutlierStrategy(Enum, str):
    WINSORIZE = auto()
    IQR = auto()
    PERCENTILE = auto()

    NONE = auto()


class ImputationStrategy(Enum, str):
    MEDIAN = auto()
    MEAN = auto()


class OutlierSingleStrategy(ABC):
    @abstractmethod
    def decide(self, skewness: float, ratio: float, is_target: bool) -> bool: ...

    @abstractmethod
    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series: ...


class IQRStrategy(OutlierSingleStrategy): ...


class WinsorizeStrategy(OutlierSingleStrategy): ...


def collect_numerical_stats(data: pd.Series) -> NumericalColumnStats:
    return NumericalColumnStats(
        max=data.max(),
        min=data.min(),
        mean=data.mean(),
        median=data.median(),
        std=data.std(),
        variance=data.var(),
        skewness=pd.to_numeric(data.skew(numeric_only=True)),
        range=np.ptp(data),
        n_unique=data.nunique(),
        quantiles={
            "p5": data.quantile(0.05),
            "p25": data.quantile(0.25),
            "p75": data.quantile(0.75),
            "p95": data.quantile(0.95),
        },
    )
