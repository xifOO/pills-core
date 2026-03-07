import csv
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from io import StringIO

from pills_core.ingestion.base import DataSource


@dataclass
class CSVOptions:
    path: str
    sep: str = ","
    encoding: str = "utf-8"
    decimal: str = "."
    thousands: str | None = None
    na_values: list[str] | None = None
    skip_rows: int = 0


class CSVDataSource(DataSource[CSVOptions]):
    def _validate(self) -> None:
        file_path = self.options.path

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if os.stat(file_path).st_size == 0:
            raise ValueError(f"File is empty: {file_path}")

    def _read_raw(self) -> str:
        with open(self.options.path, encoding=self.options.encoding) as f:
            content = f.read()

        if content.startswith("\ufeff"):
            content = content[1:]
        return content

    def _detect_separator(self, content: str) -> str:
        if self.options.sep:
            return self.options.sep

        try:
            return csv.Sniffer().sniff(content[:4096]).delimiter
        except csv.Error:
            return ","

    def _coerce_numeric(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_string_dtype(series):
            temp_col = series.astype(str).str.strip()
            clean_num = temp_col.str.replace(r"[$%\s]", "", regex=True)
            if self.options.decimal != ".":
                clean_num = clean_num.str.replace(self.options.decimal, ".", regex=False)
            
            parsed = pd.to_numeric(clean_num, errors="coerce") 
            if len(series) > 0 and parsed.notna().sum() > len(series) * 0.5:
                series = parsed.astype(np.float64)
                
        return series

    def _header_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.strip().lower() for col in df.columns]

        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate column names found: {dupes}")

        return df

    def load(self) -> pd.DataFrame:
        self._validate()

        content = self._read_raw()
        sep = self._detect_separator(content)
        df = pd.read_csv(
            StringIO(content),
            sep=sep,
            decimal=self.options.decimal,
            thousands=self.options.thousands,
            na_values=self.options.na_values,
            skiprows=self.options.skip_rows,
        )
        df = self._header_handling(df)

        for col in df.columns:
            df[col] = self._coerce_numeric(df[col])
        return df
