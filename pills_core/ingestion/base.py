from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pandas as pd


OptionsT = TypeVar("OptionsT")


class DataSource(ABC, Generic[OptionsT]):
    """
    A base class for data sources.
    This class represents a custom data source that allows for reading.
    """

    def __init__(self, options: OptionsT) -> None:
        self.options = options

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data into a pandas DataFrame."""
