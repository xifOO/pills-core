from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from pills_core._enums import DriftSeverity


@dataclass
class DriftResult:
    column_name: str
    is_drifted: bool
    pvalue: float
    psi: float
    severity: DriftSeverity
    test_used: str  # "ks" | "chi2"

    @property
    def summary(self) -> str:
        return (
            f"[{self.column_name}] drift={self.is_drifted} | "
            f"severity={self.severity} | pvalue={self.pvalue:.4f} | psi={self.psi:.4f} | "
            f"test={self.test_used}"
        )


class DriftMonitor:
    """
    Monitor for detecting Concept Drift.

    Uses statistical hypothesis testing (KS / Chi-Square) to detect
    distribution shift, and PSI to quantify its severity.
    """

    def __init__(self, critical_p_value: float = 0.01) -> None:
        self.critical_p_value = critical_p_value
        self.reference_profile: Dict[str, pd.Series] = {}

    def _get_reference(self, column_name: str) -> pd.Series:
        if column_name not in self.reference_profile:
            raise ValueError(f"Column {column_name} not found in reference profile.")
        return self.reference_profile[column_name]

    def capture_reference(self, data: pd.DataFrame) -> None:
        """
        Preserves the characteristics of the training sample.
        """
        for column in data.columns:
            if data[column].notna().any():
                self.reference_profile[column] = data[column].copy()

    def check_for_drift(self, column_name: str, current: pd.Series) -> DriftResult:
        reference = self._get_reference(column_name)

        if pd.api.types.is_numeric_dtype(reference):
            pvalue = self._ks_pvalue(reference, current)
            test_used = "ks"
        else:
            pvalue = self._chi2_pvalue(reference, current)
            test_used = "chi2"

        psi = self.calculate_psi(reference, current)

        return DriftResult(
            column_name=column_name,
            is_drifted=pvalue < self.critical_p_value,
            pvalue=pvalue,
            psi=psi,
            severity=DriftSeverity.from_psi(psi),
            test_used=test_used,
        )

    def _ks_pvalue(self, reference: pd.Series, current: pd.Series) -> float:
        result = stats.ks_2samp(current.dropna(), reference.dropna())
        return float(result.pvalue)  # type: ignore

    def _chi2_pvalue(self, reference: pd.Series, current: pd.Series) -> float:
        contingency_table = pd.crosstab(reference, current)
        _, pvalue, _, _ = stats.chi2_contingency(contingency_table)
        return float(pvalue)  # type: ignore

    def calculate_psi(
        self, reference: pd.Series, current: pd.Series, num_bins: int = 10
    ) -> float:
        """
        Alternative method Population Stability Index (PSI).
        Help to undertand, how much changed category/buckets.
        """
        if not pd.api.types.is_numeric_dtype(reference):
            ref_counts = reference.value_counts(normalize=True)
            cur_counts = current.value_counts(normalize=True)
        else:
            bins = np.unique(
                np.quantile(reference.dropna(), np.linspace(0, 1, num_bins + 1))
            ).tolist()

            if len(bins) < 2:
                return 0.0

            ref_df = pd.cut(reference, bins=bins, include_lowest=True).value_counts(
                normalize=True
            )
            cur_df = pd.cut(current, bins=bins, include_lowest=True).value_counts(
                normalize=True
            )
            ref_counts, cur_counts = ref_df, cur_df

        all_bins = ref_counts.index.union(cur_counts.index)
        ref_pct = ref_counts.reindex(all_bins, fill_value=0) + 1e-6
        cur_pct = cur_counts.reindex(all_bins, fill_value=0) + 1e-6

        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
