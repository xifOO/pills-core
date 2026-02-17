from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd


class BaseEngine(ABC):
    def __init__(self) -> None: ...

    @abstractmethod
    def fit(self, df: pd.DataFrame, roles: Dict[str, str]):
        """
        Start the training process.
        The Engine must be able to handle timeout interruptions
        and return the best model found up to that point.
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Prediction method."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model and its metadata (checkpoint)."""
        ...