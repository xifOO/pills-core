from typing import ClassVar

import pandas as pd

from pills_core._enums import FamilyRole, SemanticRole, TaskType, TransformPhase
from pills_core.strategies.base import ColumnMeta
from pills_core.strategies.numeric.base import (
    NumericalColumnMeta,
    NumericalEmbedding,
    NumericalStrategy,
)
from pills_core.types.stats import NumericalColumnStats


class NumericalImputationStrategy(NumericalStrategy):
    fills_with_existing_value: ClassVar[bool] = (
        True  # it uses central tendency or a constant
    )
    sensitive_to_outliers: ClassVar[bool] = False  # performance degrades with outliers
    sensitive_to_skewness: ClassVar[bool] = (
        False  # it's unsuitable for skewed distributions
    )
    preserves_distribution: ClassVar[bool] = (
        True  # the data shape remains relatively intact
    )
    safe_for_target: ClassVar[bool] = (
        True  # it's safe to use on the target variable (y)
    )
    requires_outliers_removed: ClassVar[bool] = False  # Mean / UpperBoundary only

    def __init__(
        self,
        *,
        embedding: NumericalEmbedding,
        radius: float,
        sensitive_outlier_ratio_limit: float = 1.0,
        sensitive_skewness_limit: float = float("inf"),
    ) -> None:
        super().__init__(embedding=embedding, radius=radius)
        self.sensitive_outlier_ratio_limit = sensitive_outlier_ratio_limit
        self.sensitive_skewness_limit = sensitive_skewness_limit

    @property
    def phase(self) -> TransformPhase:
        return TransformPhase.IMPUTATION

    def should_apply(self, stats: NumericalColumnStats, meta: ColumnMeta) -> bool:
        if not self.safe_for_target and meta.is_target:
            return False
        if (
            self.sensitive_to_outliers
            and stats.outlier_ratio > self.sensitive_outlier_ratio_limit
        ):
            return False
        if (
            self.sensitive_to_skewness
            and abs(stats.skewness) > self.sensitive_skewness_limit
        ):
            return False
        return stats.missing_ratio > 0

    def ordering_constraints(
        self, present_phases: set[TransformPhase]
    ) -> set[tuple[TransformPhase, TransformPhase]]:
        edges: set[tuple[TransformPhase, TransformPhase]] = set()

        if TransformPhase.SCALING in present_phases:
            edges.add((TransformPhase.IMPUTATION, TransformPhase.SCALING))

        if self.requires_outliers_removed and TransformPhase.OUTLIER in present_phases:
            edges.add((TransformPhase.OUTLIER, TransformPhase.IMPUTATION))

        return edges

    def explain(self, stats: NumericalColumnStats) -> str:
        parts = [f"Imputing {stats.missing_ratio:.1%} missing with '{self.name}'"]
        if not self.preserves_distribution:
            parts.append("distorts distribution")
        if self.sensitive_to_outliers:
            parts.append("sensitive to outliers")
        if self.sensitive_to_skewness:
            parts.append("sensitive to skewness")
        if not self.safe_for_target:
            parts.append("unsafe for target")
        return " | ".join(parts)


class MedianImputation(NumericalImputationStrategy):
    name: ClassVar[str] = "median"
    family_role: ClassVar[FamilyRole] = FamilyRole.CENTRAL_TENDENCY

    fills_with_existing_value: ClassVar[bool] = True
    sensitive_to_outliers: ClassVar[bool] = False
    sensitive_to_skewness: ClassVar[bool] = False
    preserves_distribution: ClassVar[bool] = True

    def is_task_valid(self, meta: NumericalColumnMeta) -> bool:
        if meta.task_type == TaskType.TIME_SERIES and meta.is_target:
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.median)


class MeanImputation(NumericalImputationStrategy):
    name: ClassVar[str] = "mean"
    family_role: ClassVar[FamilyRole] = FamilyRole.CENTRAL_TENDENCY

    fills_with_existing_value: ClassVar[bool] = True
    sensitive_to_outliers: ClassVar[bool] = True
    sensitive_to_skewness: ClassVar[bool] = True
    preserves_distribution: ClassVar[bool] = True
    requires_outliers_removed: ClassVar[bool] = True

    def is_task_valid(self, meta: NumericalColumnMeta) -> bool:
        if meta.task_type == TaskType.TIME_SERIES and meta.is_target:
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mean)


class ModeImputation(NumericalImputationStrategy):
    name: ClassVar[str] = "mode"
    family_role: ClassVar[FamilyRole] = FamilyRole.CENTRAL_TENDENCY

    fills_with_existing_value: ClassVar[bool] = True
    sensitive_to_outliers: ClassVar[bool] = False
    sensitive_to_skewness: ClassVar[bool] = False
    preserves_distribution: ClassVar[bool] = (
        False  # artificially inflates the mode frequency
    )

    def is_task_valid(self, meta: NumericalColumnMeta) -> bool:
        if meta.task_type == TaskType.TIME_SERIES and meta.is_target:
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mode)


class ZeroImputation(NumericalImputationStrategy):
    name: ClassVar[str] = "constant_zero"
    family_role: ClassVar[FamilyRole] = FamilyRole.CONSTANT

    fills_with_existing_value: ClassVar[bool] = False
    sensitive_to_outliers: ClassVar[bool] = False
    sensitive_to_skewness: ClassVar[bool] = False
    preserves_distribution: ClassVar[bool] = False

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        if meta.semantic_role in (SemanticRole.COUNT, SemanticRole.CONTINUOUS):
            return False

        if meta.domain_profile.is_monetary:
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(0)


class UpperBoundaryImputation(NumericalImputationStrategy):
    name: ClassVar[str] = "upper_boundary"
    family_role: ClassVar[FamilyRole] = FamilyRole.BOUNDARY

    fills_with_existing_value: ClassVar[bool] = True
    sensitive_to_outliers: ClassVar[bool] = True
    sensitive_to_skewness: ClassVar[bool] = False
    preserves_distribution: ClassVar[bool] = False
    safe_for_target: ClassVar[bool] = False
    requires_outliers_removed: ClassVar[bool] = True

    def __init__(
        self,
        *,
        embedding: NumericalEmbedding,
        radius: float,
        std_multiplier: float,
    ) -> None:
        super().__init__(embedding=embedding, radius=radius)
        self.std_multiplier = std_multiplier

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        if (
            meta.domain_profile.is_bounded
            and meta.domain_profile.upper_bound is not None
        ):
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mean + self.std_multiplier * stats.std)


class LowerBoundaryImputation(NumericalImputationStrategy):
    name: ClassVar[str] = "lower_boundary"
    family_role: ClassVar[FamilyRole] = FamilyRole.BOUNDARY

    fills_with_existing_value: ClassVar[bool] = True
    sensitive_to_outliers: ClassVar[bool] = True
    sensitive_to_skewness: ClassVar[bool] = False
    preserves_distribution: ClassVar[bool] = False
    safe_for_target: ClassVar[bool] = False

    def __init__(
        self,
        *,
        embedding: NumericalEmbedding,
        radius: float,
        std_multiplier: float,
    ) -> None:
        super().__init__(embedding=embedding, radius=radius)
        self.std_multiplier = std_multiplier

    def is_domain_valid(self, meta: NumericalColumnMeta) -> bool:
        if meta.domain_profile.is_monetary or meta.domain_profile.is_rate:
            return False

        if (
            meta.domain_profile.lower_bound is not None
            and meta.domain_profile.lower_bound >= 0
        ):
            return False

        return True

    def apply(self, data: pd.Series, stats: NumericalColumnStats) -> pd.Series:
        return data.fillna(stats.mean - self.std_multiplier * stats.std)
