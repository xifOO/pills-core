from typing import Any, Dict

import pandas as pd


class DriftMonitor:
    """
    Monitor for detecting Concept Drift
    """

    def __init__(self, critical_p_value: float = 0.01) -> None:
        self.critical_p_value = critical_p_value
        self.reference_profile: Dict[str, Any] = {}

    def capture_reference(self, data: pd.DataFrame):
        """
        Preserves the characteristics of the training sample.
        """
        ...

    def check_for_drift(self, column_name: str, current: pd.Series) -> bool:
        """
        Compares current data with a reference distribution.
        Uses the K-S test for continuous data.
        """
        ...

    def calculate_psi(self, reference: pd.Series, current: pd.Series) -> float:
        """
        Alternative method PSI.
        Help to undertand, how much changed category/buckets.
        """
        ...
