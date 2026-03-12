from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


class DriftMonitor:
    """
    Monitor for detecting Concept Drift
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

    def check_for_drift(self, column_name: str, current: pd.Series) -> bool:
        reference = self._get_reference(column_name)

        if pd.api.types.is_numeric_dtype(reference):
            return self._check_drift_numerical(reference, current)
        else:
            return self._check_drift_categorical(reference, current)

    def _check_drift_numerical(self, reference: pd.Series, current: pd.Series) -> bool:
        """
        Compares current data with a reference distribution.
        Uses the K-S test for numerical data.
        """
        result = stats.ks_2samp(current.dropna(), reference.dropna())
        return result.pvalue < self.critical_p_value  # type: ignore

    def _check_drift_categorical(
        self, reference: pd.Series, current: pd.Series
    ) -> bool:
        """
        Uses Chi-Square test for categorical data.
        """
        contingency_table = pd.crosstab(reference, current)
        _, pvalue, _, _ = stats.chi2_contingency(contingency_table)
        return pvalue < self.critical_p_value  # type: ignore

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

            ref_df = pd.DataFrame({"data": reference})
            ref_df["bin"] = pd.cut(ref_df["data"], bins=bins, include_lowest=True)

            cur_df = pd.DataFrame({"data": current})
            cur_df["bin"] = pd.cut(cur_df["data"], bins=bins, include_lowest=True)

            ref_counts = ref_df["bin"].value_counts(normalize=True)
            cur_counts = cur_df["bin"].value_counts(normalize=True)

        all_bins = ref_counts.index.union(cur_counts.index)
        ref_pct = ref_counts.reindex(all_bins, fill_value=0)
        cur_pct = cur_counts.reindex(all_bins, fill_value=0)

        epsilon = 1e-6  # avoid log(0)
        ref_pct = ref_pct + epsilon
        cur_pct = cur_pct + epsilon

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)
