import warnings
from typing import Literal

import pandas as pd

from pills_core.types.profiles import ColumnTypeProfile

INFER_TYPES = Literal["numeric", "categorical", "datetime", "unknown"]


class TypeInferencer:
    def __init__(
        self,
        cardinality_abs: int,
        cardinality_ratio: float,
        coercion_thresholds: float,
        max_sample_size: int,
    ) -> None:
        self.cardinality_abs = cardinality_abs
        self.cardinality_ratio = cardinality_ratio
        self.coercion_thresholds = coercion_thresholds
        self.max_sample_size = max_sample_size

    def infer(self, series: pd.Series) -> ColumnTypeProfile:
        return ColumnTypeProfile(
            name=str(series.name),
            inferred_type=self._detect_type(series),
            hints=self._build_hints(series),
        )

    def _detect_type(self, series: pd.Series) -> INFER_TYPES:
        if series.empty:
            return "unknown"

        if pd.api.types.is_bool_dtype(series):
            return "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if series.dtype == "object" or str(series.dtype).startswith("string"):
            return self._inspect_object_column(series)

        return "unknown"

    def _inspect_object_column(self, series: pd.Series) -> INFER_TYPES:
        clean = series.dropna()
        if clean.empty:
            return "unknown"

        sample = clean.sample(n=min(len(clean), self.max_sample_size), random_state=42)

        python_types = sample.map(type).unique()
        if len(python_types) > 1:
            coerced = pd.to_numeric(sample, errors="coerce")
            if coerced.notna().mean() >= self.coercion_thresholds:
                return "numeric"

            if self._looks_like_datetime(sample.astype(str)):
                return "datetime"

            warnings.warn(
                f"Column '{series.name}' contains mixed Python types and failed numeric coercion. "
                f"Returning 'unknown'. Consider cleaning the source or using schema overrides.",
                stacklevel=5,
            )
            return "unknown"

        cardinality = clean.nunique()
        ratio = cardinality / len(clean)

        if cardinality < self.cardinality_abs or ratio < self.cardinality_ratio:
            return "categorical"

        if self._looks_like_datetime(sample):
            return "datetime"

        return "unknown"

    def _looks_like_datetime(self, series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series):
            return False

        try:
            coerced = pd.to_datetime(series, errors="coerce")
            return coerced.notna().mean() >= self.coercion_thresholds
        except Exception:
            return False

    def _build_hints(self, series: pd.Series) -> dict:
        if series.empty:
            return {"cardinality": 0, "missing_rate": 0.0, "dtype": str(series.dtype)}

        clean = series.dropna()
        sample = (
            clean.sample(min(1_000, len(clean)), random_state=42)
            if len(clean) > 0
            else clean
        )
        return {
            "cardinality": sample.nunique(),
            "missing_rate": float(series.isna().mean()),
            "dtype": str(series.dtype),
        }
