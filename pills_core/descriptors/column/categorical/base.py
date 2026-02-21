import pandas as pd
from pills_core.descriptors.column.column import ColumnDescriptor
from pills_core.types.profiles import Cardinality, CategoricalProfile


class BaseCategoricalDescriptor(ColumnDescriptor):
    RARE_THRESHOLD = 0.01
    LOW_CARD_MAX = 10
    MEDIUM_CARD_MAX = 100

    def profile(self, data: pd.Series) -> CategoricalProfile:
        n_unique = data.nunique()
        return CategoricalProfile(
            cardinality=self._cardinality(n_unique),
            n_unique=n_unique,
            rare_categories=self._rare_categories(data),
            has_typos=self._has_typos(data),
        )

    def _cardinality(self, n_unique: int) -> Cardinality:
        if n_unique <= self.LOW_CARD_MAX:
            return Cardinality.LOW
        if n_unique <= self.MEDIUM_CARD_MAX:
            return Cardinality.MEDIUM
        return Cardinality.HIGH

    def _rare_categories(self, data: pd.Series) -> list[str]:
        freq = data.value_counts(normalize=True)
        return freq[freq < self.RARE_THRESHOLD].index.tolist()

    def _has_typos(self, data: pd.Series) -> bool:
        """Detect case/whitespace inconsistencies."""
        values = data.dropna().astype(str)
        normalized = values.str.lower().str.strip()
        return normalized.nunique() < values.nunique()

    def _normalize(self, data: pd.Series) -> pd.Series:
        """Fix case/whitespace before any further transformation."""
        return data.astype(str).str.lower().str.strip()
