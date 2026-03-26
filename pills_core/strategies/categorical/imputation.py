from typing import ClassVar

from pills_core._enums import TransformPhase
from pills_core.strategies.categorical.base import (
    CategoricalColumnMeta,
    CategoricalStrategy,
)
from pills_core.types.profiles import Cardinality
from pills_core.types.stats import CategoricalColumnStats


class CategoricalImputationStrategy(CategoricalStrategy):
    fills_with_existing_value: ClassVar[bool] = True  # uses existing category
    creates_new_category: ClassVar[bool] = False  # add a separate missing category
    sensitive_to_imbalance: ClassVar[bool] = True  # bad for unbalance
    sensitive_to_high_cardinality: ClassVar[bool] = False  # bad for high cardinality
    sensitive_to_rare_categories: ClassVar[bool] = False  # delete rare categories
    preserves_distribution: ClassVar[bool] = True
    is_deterministic: ClassVar[bool] = True  # always the same result
    handles_rare_categories: ClassVar[bool] = False  # correctly handles rare categories
    safe_for_target: ClassVar[bool] = True
    requires_cleaning: ClassVar[bool] = False
    requires_encoding: ClassVar[bool] = False
    supports_ordinal: ClassVar[bool] = False

    @property
    def phase(self) -> TransformPhase:
        return TransformPhase.IMPUTATION

    def should_apply(
        self, stats: CategoricalColumnStats, meta: CategoricalColumnMeta
    ) -> bool:
        if not self.safe_for_target and meta.is_target:
            return False

        if (
            self.sensitive_to_high_cardinality
            and meta.profile.cardinality == Cardinality.HIGH
        ):
            return False

        if self.sensitive_to_rare_categories and meta.profile.rare_categories:
            return False

        if self.requires_cleaning and meta.profile.has_typos:
            return False

        if self.supports_ordinal and not meta.profile.has_order:
            return False

        return True

    def explain(self, stats: CategoricalColumnStats) -> str:
        parts = [f"Imputing categorical column with '{self.__class__.__name__}'"]
        if not self.preserves_distribution:
            parts.append("distorts category distribution")
        if self.sensitive_to_high_cardinality:
            parts.append("sensitive to high cardinality")
        if self.sensitive_to_rare_categories:
            parts.append("sensitive to rare categories")
        if not self.safe_for_target:
            parts.append("unsafe for target")
        if self.requires_cleaning:
            parts.append("requires cleaning for typos")
        return " | ".join(parts)
