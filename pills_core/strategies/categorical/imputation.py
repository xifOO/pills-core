from typing import ClassVar

import pandas as pd

from pills_core._enums import TaskType, TransformPhase
from pills_core.explain import Explanation
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

    def explain(
        self, stats: CategoricalColumnStats, meta: CategoricalColumnMeta
    ) -> Explanation:
        reasons = [f"Imputing categorical column with '{self.__class__.__name__}'"]
        if not self.preserves_distribution:
            reasons.append("distorts category distribution")
        if self.sensitive_to_high_cardinality:
            reasons.append("sensitive to high cardinality")
        if self.sensitive_to_rare_categories:
            reasons.append("sensitive to rare categories")
        if not self.safe_for_target:
            reasons.append("unsafe for target")
        if self.requires_cleaning:
            reasons.append("requires cleaning for typos")

        return Explanation(
            name=self.name,
            value="selected" if self.should_apply(stats, meta) else "rejected",
            reasons=reasons,
        )


class MostFrequentStrategy(CategoricalImputationStrategy):
    def is_task_valid(self, meta: CategoricalColumnMeta) -> bool:
        if (
            meta.task_type == TaskType.REGRESSION
            or meta.task_type == TaskType.TIME_SERIES
        ):
            return False

        return True

    def is_domain_valid(self, meta: CategoricalColumnMeta) -> bool:
        if meta.profile.has_order:
            return False

        if meta.profile.cardinality == Cardinality.HIGH and meta.profile.n_unique > 100:
            return False

        return True

    def should_apply(
        self, stats: CategoricalColumnStats, meta: CategoricalColumnMeta
    ) -> bool:
        if stats.missing_ratio > 0.1:
            return False

        if stats.most_frequent_ratio < 0.2:
            return False

        if stats.rare_ratio > 0.3:
            return False

        return super().should_apply(stats, meta)

    def apply(self, data: pd.Series, stats: CategoricalColumnStats) -> pd.Series:
        return data.fillna(stats.mode)


class MissingStrategy(CategoricalImputationStrategy):
    creates_new_category: ClassVar[bool] = True
    sensitive_to_imbalance: ClassVar[bool] = False

    def is_domain_valid(self, meta: CategoricalColumnMeta) -> bool:
        if meta.profile.cardinality == Cardinality.HIGH:
            return False

        return True

    def should_apply(
        self, stats: CategoricalColumnStats, meta: CategoricalColumnMeta
    ) -> bool:
        if stats.n_unique > 3 and stats.missing_ratio < 0.1 and stats.rare_ratio < 0.2:
            return False

        if stats.rare_ratio < 0.2 and stats.missing_ratio < 0.05:
            return False

        if stats.missing_ratio < 0.05:
            if not (
                meta.profile.is_domain_specific
                or meta.profile.has_order
                or stats.n_unique <= 3
            ):
                return False

        return super().should_apply(stats, meta)

    def apply(self, data: pd.Series, stats: CategoricalColumnStats) -> pd.Series:
        return data.fillna("__MISSING__")


class ForwardFillStrategy(CategoricalImputationStrategy):
    sensitive_to_imbalance: ClassVar[bool] = False
    preserves_distribution: ClassVar[bool] = False
    handles_rare_categories: ClassVar[bool] = True
    safe_for_target: ClassVar[bool] = False
    supports_ordinal: ClassVar[bool] = True

    def is_task_valid(self, meta: CategoricalColumnMeta) -> bool:
        return meta.task_type == TaskType.TIME_SERIES

    def is_domain_valid(self, meta: CategoricalColumnMeta) -> bool:
        return meta.profile.has_order

    def should_apply(
        self, stats: CategoricalColumnStats, meta: CategoricalColumnMeta
    ) -> bool:
        if stats.missing_ratio > 0.5:
            return False

        if meta.profile.has_leading_nulls:
            return False

        return super().should_apply(stats, meta)

    def apply(self, data: pd.Series, stats: CategoricalColumnStats) -> pd.Series:
        return data.ffill()


class BackwardFillStrategy(CategoricalImputationStrategy):
    sensitive_to_imbalance: ClassVar[bool] = False
    preserves_distribution: ClassVar[bool] = False
    handles_rare_categories: ClassVar[bool] = True
    safe_for_target: ClassVar[bool] = False
    supports_ordinal: ClassVar[bool] = True

    def is_task_valid(self, meta: CategoricalColumnMeta) -> bool:
        return meta.task_type == TaskType.TIME_SERIES

    def is_domain_valid(self, meta: CategoricalColumnMeta) -> bool:
        return meta.profile.has_order

    def should_apply(
        self, stats: CategoricalColumnStats, meta: CategoricalColumnMeta
    ) -> bool:
        if stats.missing_ratio > 0.5:
            return False

        if stats.missing_ratio < 0.1 and not meta.profile.has_leading_nulls:
            return False

        return super().should_apply(stats, meta)

    def apply(self, data: pd.Series, stats: CategoricalColumnStats) -> pd.Series:
        return data.bfill()
