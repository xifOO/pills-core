from pathlib import Path
from typing import Dict, Any, Final
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pills_core._enums import TaskType
from loguru import logger
from config import config


class BaseInspector(ABC):
    @abstractmethod
    def scan(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main method: assigns a role to each column.
        Logic should include:
        - If a column has many unique strings -> ColumnRole.ID
        - If a column has 95% missing values -> ColumnRole.DROP
        """
        ...

    @abstractmethod
    def get_task_type(self, df: pd.DataFrame, target: str) -> TaskType:
        """
        Auto-detection: if target is float -> REGRESSION,
        if 2 unique values -> BINARY.
        """
        ...


class Inspector:
    bool_map: Final = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
    }

    def __init__(self, file_path: Path) -> None:
        logger.info(f"Loading data from {file_path}")
        self.df = pd.read_csv(file_path)
        self.raw_df = self.df.copy()

    def _smart_type_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            temp_col = df[col].astype(str).str.strip()

            clean_num = pd.to_numeric(
                temp_col.str.replace(r"[$,%\s]", "", regex=True).str.replace(
                    ",", ".", regex=False
                ),
                errors="coerce",
            )
            if clean_num.notna().sum() > len(df) * 0.5:
                df[col] = clean_num.astype(np.float64)
                continue

            if config.data.date_features:
                try:
                    parsed_dates = pd.to_datetime(temp_col, errors="coerce")
                    if parsed_dates.notna().sum() > len(df) * 0.7:
                        df[col] = parsed_dates
                        continue
                except:
                    pass

            lower = temp_col.str.lower()
            if lower.isin(self.bool_map.keys()).mean() > 0.9:
                df[col] = lower.map(self.bool_map)
                continue

            if df[col].nunique() < config.data.max_unique_categories:
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(str)

        return df

    def _drop_uninformative_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rows_count = len(df)
        target_col = config.target if config.target else df.columns[-1]

        cols_to_drop = []

        for col in df.columns:
            if col == target_col:
                continue

            if df[col].nunique() == rows_count:
                cols_to_drop.append(col)

            elif df[col].nunique() <= 1:
                cols_to_drop.append(col)

        return df.drop(columns=cols_to_drop)

    def scan(self):
        self.df = self._smart_type_inference(self.df)
        if config.data.drop_uninformative:
            self.df = self._drop_uninformative_columns(self.df)
        # _handle_missing  _remove_outliers _expand_date_features
        if "age" in self.df.columns:
            self.df.loc[self.df["age"] > 120, "age"] = np.nan
            logger.warning("Outliers removed in 'age'")

    def get_data_for_engine(self):
        target_col = self.df.columns[-1]

        x = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        return x, y


path = Path("bad_data.csv")
inspector = Inspector(path)
inspector.scan()
x, y = inspector.get_data_for_engine()

print(x)
print(x.dtypes)
