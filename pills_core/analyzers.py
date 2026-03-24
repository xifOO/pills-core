from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Mapping, Type, TypeVar

import pandas as pd

from pills_core._enums import ColumnRole, SemanticRole, TaskType
from pills_core.analyzer_rules import (
    Decision,
    DomainPolicy,
    DomainRule,
    DomainTags,
    MatchPolicy,
    MatchRule,
)
from pills_core.strategies.base import ColumnMeta, StrategyEmbedding
from pills_core.types.profiles import (
    Cardinality,
    CategoricalProfile,
    DomainProfile,
    StatisticalProfile,
)
from pills_core.types.stats import CategoricalColumnStats, NumericalColumnStats

StatsT = TypeVar("StatsT", NumericalColumnStats, CategoricalColumnStats)


@dataclass(frozen=True, slots=True)
class NumericalThresholds:
    id_unique_ratio: float = 0.95
    id_monotonic_ratio: float = 0.95
    binary_max_unique: int = 2
    low_unique_ratio: float = 0.15
    low_unique_abs: int = 20
    ordinal_max_skewness: float = 1.0
    count_skewness: float = 0.5
    skewed_threshold: float = 1.0
    heavy_tail_kurtosis: float = 3.0
    sparse_missing_ratio: float = 0.3
    low_variance_cv: float = 0.01


@dataclass(frozen=True, slots=True)
class CategoricalThresholds:
    binary_max_unique: int = 2
    low_cardinality_max: int = 10
    medium_cardinality_max: int = 100
    rare_ratio_threshold: float = 0.05
    dominant_ratio_threshold: float = 0.95
    sparse_missing_ratio: float = 0.3


@dataclass(frozen=True, slots=True)
class DomainRuleConfig:
    name: str
    keywords: tuple[str, ...]
    tags: DomainTags


@dataclass(frozen=True, slots=True)
class DomainConfig:
    rules: tuple[DomainRuleConfig, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    numerical: NumericalThresholds = field(default_factory=NumericalThresholds)
    categorical: CategoricalThresholds = field(default_factory=CategoricalThresholds)
    domain: DomainConfig = field(default_factory=DomainConfig)


@dataclass(frozen=True, slots=True)
class NumericalRuleContext:
    stats: NumericalColumnStats


@dataclass(frozen=True, slots=True)
class CategoricalRuleContext:
    stats: CategoricalColumnStats


def build_analyzer_config(payload: Mapping[str, Any]) -> AnalyzerConfig:
    numerical_payload = payload.get("numerical", {})
    categorical_payload = payload.get("categorical", {})
    domain_payload = payload.get("domain", {})

    if "thresholds" in numerical_payload:
        numerical_payload = numerical_payload["thresholds"]
    if "thresholds" in categorical_payload:
        categorical_payload = categorical_payload["thresholds"]

    return AnalyzerConfig(
        numerical=NumericalThresholds(**numerical_payload),
        categorical=CategoricalThresholds(**categorical_payload),
        domain=build_domain_config(domain_payload),
    )


def build_domain_config(payload: Mapping[str, Any]) -> DomainConfig:
    rules_payload = payload.get("rules")
    if rules_payload is not None:
        return DomainConfig(
            rules=tuple(build_domain_rule_config(rule) for rule in rules_payload)
        )

    keywords_payload = payload.get("keywords", payload)
    if not isinstance(keywords_payload, Mapping):
        return DomainConfig()

    generated_rules = []
    for name, keywords in keywords_payload.items():
        generated_rules.append(
            DomainRuleConfig(
                name=str(name),
                keywords=tuple(str(keyword) for keyword in keywords),
                tags=DomainTags(**{f"is_{name}": True}),
            )
        )
    return DomainConfig(rules=tuple(generated_rules))


def build_domain_rule_config(payload: Mapping[str, Any]) -> DomainRuleConfig:
    tags_payload = payload.get("tags", {})
    return DomainRuleConfig(
        name=str(payload["name"]),
        keywords=tuple(str(keyword) for keyword in payload.get("keywords", ())),
        tags=DomainTags(
            is_ratio=bool(tags_payload.get("is_ratio", False)),
            is_monetary=bool(tags_payload.get("is_monetary", False)),
            is_rate=bool(tags_payload.get("is_rate", False)),
            is_score=bool(tags_payload.get("is_score", False)),
        ),
    )


def default_domain_config() -> DomainConfig:
    return DomainConfig(
        rules=(
            DomainRuleConfig(
                name="ratio",
                keywords=("ratio", "share", "util", "pct", "percent"),
                tags=DomainTags(is_ratio=True),
            ),
            DomainRuleConfig(
                name="monetary",
                keywords=(
                    "amnt",
                    "amount",
                    "bal",
                    "balance",
                    "limit",
                    "installment",
                    "recoveries",
                    "fee",
                    "payment",
                    "pymnt",
                    "prncp",
                    "inv",
                ),
                tags=DomainTags(is_monetary=True),
            ),
            DomainRuleConfig(
                name="rate",
                keywords=(
                    "int_rate",
                    "revol_util",
                    "il_util",
                    "bc_util",
                    "all_util",
                    "sec_app_revol_util",
                    "pct",
                    "percent",
                    "dti",
                    "rate",
                ),
                tags=DomainTags(is_rate=True),
            ),
            DomainRuleConfig(
                name="score",
                keywords=("fico", "grade", "score"),
                tags=DomainTags(is_score=True),
            ),
        )
    )


def default_analyzer_config() -> AnalyzerConfig:
    return AnalyzerConfig(domain=default_domain_config())


def build_domain_policy(config: DomainConfig) -> DomainPolicy:
    return DomainPolicy(
        rules=tuple(
            DomainRule(name=rule.name, keywords=rule.keywords, tags=rule.tags)
            for rule in config.rules
        )
    )


def build_numerical_semantic_policy(
    thresholds: NumericalThresholds,
) -> MatchPolicy[NumericalRuleContext, SemanticRole]:
    return MatchPolicy(
        rules=(
            MatchRule(
                name="binary",
                value=SemanticRole.BINARY,
                predicate=lambda ctx: ctx.stats.n_unique <= thresholds.binary_max_unique,
                reason_builder=lambda ctx: (
                    (
                        f"n_unique={ctx.stats.n_unique} <= "
                        f"binary_max_unique={thresholds.binary_max_unique}."
                    ),
                ),
            ),
            MatchRule(
                name="id_like",
                value=SemanticRole.ID_LIKE,
                predicate=lambda ctx: (
                    ctx.stats.unique_ratio >= thresholds.id_unique_ratio
                    and ctx.stats.monotonic_ratio >= thresholds.id_monotonic_ratio
                ),
                reason_builder=lambda ctx: (
                    "High uniqueness and near-monotonic ordering indicate identifier-like values.",
                    (
                        f"unique_ratio={ctx.stats.unique_ratio:.3f}, "
                        f"monotonic_ratio={ctx.stats.monotonic_ratio:.3f}."
                    ),
                ),
            ),
            MatchRule(
                name="count_positive_skew",
                value=SemanticRole.COUNT,
                predicate=lambda ctx: (
                    ctx.stats.is_integer_valued
                    and ctx.stats.min >= 0
                    and ctx.stats.skewness >= thresholds.count_skewness
                    and ctx.stats.zero_ratio < 0.95
                ),
                reason_builder=lambda ctx: (
                    "Integer non-negative values with positive skew are count-like.",
                    (
                        f"skewness={ctx.stats.skewness:.3f} >= "
                        f"count_skewness={thresholds.count_skewness:.3f}."
                    ),
                ),
            ),
            MatchRule(
                name="count_sparse_zero",
                value=SemanticRole.COUNT,
                predicate=lambda ctx: (
                    ctx.stats.is_integer_valued
                    and ctx.stats.min >= 0
                    and ctx.stats.zero_ratio >= 0.95
                    and ctx.stats.n_unique <= 10
                ),
                reason_builder=lambda ctx: (
                    "Mostly-zero sparse integer values are treated as counts.",
                    f"zero_ratio={ctx.stats.zero_ratio:.3f}, n_unique={ctx.stats.n_unique}.",
                ),
            ),
            MatchRule(
                name="ordinal",
                value=SemanticRole.ORDINAL,
                predicate=lambda ctx: (
                    ctx.stats.is_integer_valued
                    and ctx.stats.unique_ratio <= thresholds.low_unique_ratio
                    and ctx.stats.n_unique <= thresholds.low_unique_abs
                    and abs(ctx.stats.skewness) <= thresholds.ordinal_max_skewness
                ),
                reason_builder=lambda ctx: (
                    "Low-cardinality integer values with moderate skew are treated as ordinal.",
                    (
                        f"unique_ratio={ctx.stats.unique_ratio:.3f}, "
                        f"abs(skewness)={abs(ctx.stats.skewness):.3f}."
                    ),
                ),
            ),
            MatchRule(
                name="numeric_nominal",
                value=SemanticRole.NUMERIC_NOMINAL,
                predicate=lambda ctx: (
                    ctx.stats.is_integer_valued
                    and ctx.stats.unique_ratio <= thresholds.low_unique_ratio
                    and ctx.stats.n_unique <= thresholds.low_unique_abs
                ),
                reason_builder=lambda ctx: (
                    "Low-cardinality integers with stronger skew are treated as nominal codes.",
                    f"abs(skewness)={abs(ctx.stats.skewness):.3f}.",
                ),
            ),
        ),
        fallback_value=SemanticRole.CONTINUOUS,
        fallback_reasons=(
            "Fallback numerical semantic role.",
            "Column is not binary, id-like, count-like, or low-cardinality integer.",
        ),
    )


def build_numerical_task_policy(
    thresholds: NumericalThresholds,
) -> MatchPolicy[NumericalRuleContext, TaskType]:
    return MatchPolicy(
        rules=(
            MatchRule(
                name="binary_target",
                value=TaskType.BINARY,
                predicate=lambda ctx: (
                    ctx.stats.n_unique == 2 and ctx.stats.is_integer_valued
                ),
                reason_builder=lambda ctx: (
                    "Target has exactly two integer-coded classes.",
                ),
            ),
            MatchRule(
                name="multiclass_target",
                value=TaskType.MULTICLASS,
                predicate=lambda ctx: (
                    ctx.stats.unique_ratio <= thresholds.low_unique_ratio
                    and ctx.stats.is_integer_valued
                    and ctx.stats.n_unique <= thresholds.low_unique_abs
                ),
                reason_builder=lambda ctx: (
                    "Low-cardinality integer target is treated as multiclass.",
                    (
                        f"n_unique={ctx.stats.n_unique}, "
                        f"unique_ratio={ctx.stats.unique_ratio:.3f}."
                    ),
                ),
            ),
        ),
        fallback_value=TaskType.REGRESSION,
        fallback_reasons=("Fallback numerical task type.",),
    )


def build_categorical_semantic_policy(
    thresholds: CategoricalThresholds,
) -> MatchPolicy[CategoricalRuleContext, SemanticRole]:
    return MatchPolicy(
        rules=(
            MatchRule(
                name="binary",
                value=SemanticRole.BINARY,
                predicate=lambda ctx: ctx.stats.n_unique <= thresholds.binary_max_unique,
                reason_builder=lambda ctx: (
                    (
                        f"n_unique={ctx.stats.n_unique} <= "
                        f"binary_max_unique={thresholds.binary_max_unique}."
                    ),
                ),
            ),
            MatchRule(
                name="ordinal_like",
                value=SemanticRole.ORDINAL,
                predicate=lambda ctx: (
                    ctx.stats.n_unique <= thresholds.low_cardinality_max
                    and ctx.stats.rare_ratio <= thresholds.rare_ratio_threshold
                    and ctx.stats.most_frequent_ratio
                    < thresholds.dominant_ratio_threshold
                ),
                reason_builder=lambda ctx: (
                    "Low-cardinality categories with balanced frequency look ordinal-like.",
                    (
                        f"rare_ratio={ctx.stats.rare_ratio:.3f}, "
                        f"most_frequent_ratio={ctx.stats.most_frequent_ratio:.3f}."
                    ),
                ),
            ),
            MatchRule(
                name="nominal_low_cardinality",
                value=SemanticRole.NUMERIC_NOMINAL,
                predicate=lambda ctx: ctx.stats.n_unique <= thresholds.low_cardinality_max,
                reason_builder=lambda ctx: (
                    "Low-cardinality categories are treated as nominal by default.",
                    (
                        f"rare_ratio={ctx.stats.rare_ratio:.3f}, "
                        f"most_frequent_ratio={ctx.stats.most_frequent_ratio:.3f}."
                    ),
                ),
            ),
        ),
        fallback_value=SemanticRole.NUMERIC_NOMINAL,
        fallback_reasons=(
            "Higher-cardinality categorical values default to nominal.",
        ),
    )


def build_categorical_task_policy(
    thresholds: CategoricalThresholds,
) -> MatchPolicy[CategoricalRuleContext, TaskType]:
    return MatchPolicy(
        rules=(
            MatchRule(
                name="binary_target",
                value=TaskType.BINARY,
                predicate=lambda ctx: ctx.stats.n_unique <= thresholds.binary_max_unique,
                reason_builder=lambda ctx: ("Target has two categories.",),
            ),
        ),
        fallback_value=TaskType.MULTICLASS,
        fallback_reasons=(
            "Categorical targets with more than two categories are multiclass.",
        ),
    )


class DomainProfiler:
    def build_for_numerical(
        self,
        tags: DomainTags,
        stats: NumericalColumnStats,
    ) -> DomainProfile:
        is_stat_bounded = (
            stats.min >= 0
            and stats.max <= 1.0
            and not tags.is_monetary
            and not tags.is_score
        )
        is_bounded = tags.is_ratio or tags.is_rate or tags.is_score or is_stat_bounded
        lower_bound = 0.0 if tags.is_monetary or is_bounded else None

        if tags.is_rate or tags.is_ratio:
            upper_bound = 100.0 if stats.max > 1.0 else 1.0
        else:
            upper_bound = None

        return DomainProfile(
            is_ratio=tags.is_ratio,
            is_monetary=tags.is_monetary,
            is_rate=tags.is_rate,
            is_score=tags.is_score,
            is_bounded=is_bounded,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def build_for_categorical(self, tags: DomainTags) -> DomainProfile:
        return DomainProfile(
            is_ratio=tags.is_ratio,
            is_monetary=tags.is_monetary,
            is_rate=tags.is_rate,
            is_score=tags.is_score,
            is_bounded=False,
            lower_bound=None,
            upper_bound=None,
        )


class ColumnAnalyzer(ABC, Generic[StatsT]):
    _registered_types: ClassVar[tuple[Type["ColumnAnalyzer[Any]"], ...]] = ()

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        domain_policy: DomainPolicy | None = None,
        domain_profiler: DomainProfiler | None = None,
    ) -> None:
        self.config = config or default_analyzer_config()
        self.domain_policy = domain_policy or build_domain_policy(self.config.domain)
        self.domain_profiler = domain_profiler or DomainProfiler()

    def __init_subclass__(cls, *, register: bool = True, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if register and not inspect.isabstract(cls):
            ColumnAnalyzer._registered_types = (
                *ColumnAnalyzer._registered_types,
                cls,
            )

    @classmethod
    def registered_types(cls) -> tuple[type["ColumnAnalyzer[Any]"], ...]:
        return cls._registered_types

    @classmethod
    def create_registered(
        cls,
        config: AnalyzerConfig | None = None,
        domain_policy: DomainPolicy | None = None,
        domain_profiler: DomainProfiler | None = None,
    ) -> tuple["ColumnAnalyzer[Any]", ...]:
        return tuple(
            analyzer_type(
                config=config,
                domain_policy=domain_policy,
                domain_profiler=domain_profiler,
            )
            for analyzer_type in cls.registered_types()
        )

    @property
    @abstractmethod
    def column_role(self) -> ColumnRole: ...

    @abstractmethod
    def can_analyze(self, series: pd.Series) -> bool: ...

    @abstractmethod
    def resolve_semantic_role(self, stats: StatsT) -> Decision[SemanticRole]: ...

    @abstractmethod
    def explain(self, stats: StatsT) -> list[str]: ...

    @abstractmethod
    def build_statistical_profile(self, stats: StatsT) -> StatisticalProfile: ...

    @abstractmethod
    def build_column_embedding(
        self,
        stats: StatsT,
        meta: ColumnMeta,
    ) -> StrategyEmbedding: ...

    @abstractmethod
    def build_meta(
        self,
        series: pd.Series,
        stats: StatsT,
        is_target: bool,
        task_type: TaskType = TaskType.AUTO,
    ) -> ColumnMeta: ...

    @abstractmethod
    def _infer_task_type(self, stats: StatsT) -> Decision[TaskType]: ...

    def detect_column_role(self, series: pd.Series) -> ColumnRole:
        if self.can_analyze(series):
            return self.column_role
        return ColumnRole.DROP

    def detect_semantic_role(self, stats: StatsT) -> SemanticRole:
        return self.resolve_semantic_role(stats).value

    def resolve_task_type(self, stats: StatsT, task_type: TaskType) -> TaskType:
        if task_type == TaskType.AUTO:
            return self._infer_task_type(stats).value
        return task_type

    def explain_task_type(self, stats: StatsT, task_type: TaskType) -> list[str]:
        if task_type != TaskType.AUTO:
            return [f"Task type provided explicitly: {task_type.value}."]
        decision = self._infer_task_type(stats)
        return [f"Resolved task type: {decision.value.value}.", *decision.reasons]


class NumericalColumnAnalyzer(ColumnAnalyzer[NumericalColumnStats]):
    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        domain_policy: DomainPolicy | None = None,
        domain_profiler: DomainProfiler | None = None,
        semantic_policy: MatchPolicy[NumericalRuleContext, SemanticRole] | None = None,
        task_policy: MatchPolicy[NumericalRuleContext, TaskType] | None = None,
    ) -> None:
        super().__init__(
            config=config,
            domain_policy=domain_policy,
            domain_profiler=domain_profiler,
        )
        self.semantic_policy = semantic_policy or build_numerical_semantic_policy(
            self.config.numerical
        )
        self.task_policy = task_policy or build_numerical_task_policy(
            self.config.numerical
        )

    @property
    def column_role(self) -> ColumnRole:
        return ColumnRole.NUMERICAL

    def can_analyze(self, series: pd.Series) -> bool:
        return bool(pd.api.types.is_numeric_dtype(series))

    def resolve_semantic_role(
        self,
        stats: NumericalColumnStats,
    ) -> Decision[SemanticRole]:
        return self.semantic_policy.resolve(NumericalRuleContext(stats=stats))

    def explain(self, stats: NumericalColumnStats) -> list[str]:
        decision = self.resolve_semantic_role(stats)
        return [f"Resolved semantic role: {decision.value.name}.", *decision.reasons]

    def build_statistical_profile(
        self,
        stats: NumericalColumnStats,
    ) -> StatisticalProfile:
        thresholds = self.config.numerical
        return StatisticalProfile(
            is_skewed=bool(abs(stats.skewness) >= thresholds.skewed_threshold),
            is_heavy_tailed=bool(stats.kurtosis > thresholds.heavy_tail_kurtosis),
            has_outliers=stats.outlier_ratio > 0,
            is_sparse=stats.missing_ratio >= thresholds.sparse_missing_ratio,
            is_low_variance=stats.cv < thresholds.low_variance_cv,
        )

    def build_column_embedding(
        self,
        stats: NumericalColumnStats,
        meta: ColumnMeta,
    ) -> StrategyEmbedding:
        skewness_sensitivity = min(abs(stats.skewness) / 3.0, 1.0)
        outliers_sensitivity = min(stats.outlier_ratio * 5.0, 1.0)

        distribution_preservation = 1.0
        if meta.profile.is_skewed:
            distribution_preservation -= 0.4
        if meta.profile.is_heavy_tailed:
            distribution_preservation -= 0.3
        distribution_preservation = max(distribution_preservation, 0.0)

        return StrategyEmbedding(
            skewness_sensitivity=skewness_sensitivity,
            outliers_sensitivity=outliers_sensitivity,
            missing_ratio_fit=1.0 if stats.missing_ratio > 0 else 0.0,
            distribution_preservation=distribution_preservation,
            target_safety=1.0 if meta.is_target else 0.0,
            cardinality_fit=1.0 - min(stats.unique_ratio, 1.0),
        )

    def build_meta(
        self,
        series: pd.Series,
        stats: NumericalColumnStats,
        is_target: bool,
        task_type: TaskType = TaskType.AUTO,
    ) -> ColumnMeta:
        domain_tags = self.domain_policy.resolve(str(series.name))
        return ColumnMeta(
            role=self.detect_column_role(series),
            semantic_role=self.detect_semantic_role(stats),
            profile=self.build_statistical_profile(stats),
            is_target=is_target,
            domain_profile=self.domain_profiler.build_for_numerical(domain_tags, stats),
            task_type=self.resolve_task_type(stats, task_type),
        )

    def _infer_task_type(self, stats: NumericalColumnStats) -> Decision[TaskType]:
        return self.task_policy.resolve(NumericalRuleContext(stats=stats))


class CategoricalColumnAnalyzer(ColumnAnalyzer[CategoricalColumnStats]):
    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        domain_policy: DomainPolicy | None = None,
        domain_profiler: DomainProfiler | None = None,
        semantic_policy: MatchPolicy[CategoricalRuleContext, SemanticRole] | None = None,
        task_policy: MatchPolicy[CategoricalRuleContext, TaskType] | None = None,
    ) -> None:
        super().__init__(
            config=config,
            domain_policy=domain_policy,
            domain_profiler=domain_profiler,
        )
        self.semantic_policy = semantic_policy or build_categorical_semantic_policy(
            self.config.categorical
        )
        self.task_policy = task_policy or build_categorical_task_policy(
            self.config.categorical
        )

    @property
    def column_role(self) -> ColumnRole:
        return ColumnRole.CATEGORICAL

    def can_analyze(self, series: pd.Series) -> bool:
        return not bool(pd.api.types.is_numeric_dtype(series))

    def resolve_semantic_role(
        self,
        stats: CategoricalColumnStats,
    ) -> Decision[SemanticRole]:
        return self.semantic_policy.resolve(CategoricalRuleContext(stats=stats))

    def explain(self, stats: CategoricalColumnStats) -> list[str]:
        decision = self.resolve_semantic_role(stats)
        return [f"Resolved semantic role: {decision.value.name}.", *decision.reasons]

    def build_statistical_profile(
        self,
        stats: CategoricalColumnStats,
    ) -> StatisticalProfile:
        thresholds = self.config.categorical
        return StatisticalProfile(
            is_skewed=stats.entropy < 1.0,
            is_heavy_tailed=False,
            has_outliers=bool(stats.rare_categories),
            is_sparse=stats.missing_ratio >= thresholds.sparse_missing_ratio,
            is_low_variance=(
                stats.most_frequent_ratio >= thresholds.dominant_ratio_threshold
            ),
        )

    def build_categorical_profile(
        self,
        stats: CategoricalColumnStats,
        domain_tags: DomainTags,
    ) -> CategoricalProfile:
        thresholds = self.config.categorical

        if stats.n_unique <= thresholds.low_cardinality_max:
            cardinality = Cardinality.LOW
        elif stats.n_unique <= thresholds.medium_cardinality_max:
            cardinality = Cardinality.MEDIUM
        else:
            cardinality = Cardinality.HIGH

        return CategoricalProfile(
            cardinality=cardinality,
            n_unique=int(stats.n_unique),
            rare_categories=list(stats.rare_categories),
            has_typos=False,
            has_order=self.detect_semantic_role(stats) is SemanticRole.ORDINAL,
            is_domain_specific=any(
                (
                    domain_tags.is_ratio,
                    domain_tags.is_monetary,
                    domain_tags.is_rate,
                    domain_tags.is_score,
                )
            ),
        )

    def build_column_embedding(
        self,
        stats: CategoricalColumnStats,
        meta: ColumnMeta,
    ) -> StrategyEmbedding:
        thresholds = self.config.categorical
        return StrategyEmbedding(
            skewness_sensitivity=min(1.0 - min(stats.entropy / 5.0, 1.0), 1.0),
            outliers_sensitivity=min(stats.rare_ratio * 5.0, 1.0),
            missing_ratio_fit=min(stats.missing_ratio * 2.0, 1.0),
            distribution_preservation=1.0 if not meta.profile.is_low_variance else 0.5,
            target_safety=1.0 if meta.is_target else 0.2,
            cardinality_fit=min(
                stats.n_unique / thresholds.medium_cardinality_max,
                1.0,
            ),
        )

    def build_meta(
        self,
        series: pd.Series,
        stats: CategoricalColumnStats,
        is_target: bool,
        task_type: TaskType = TaskType.AUTO,
    ) -> ColumnMeta:
        domain_tags = self.domain_policy.resolve(str(series.name))
        return ColumnMeta(
            role=self.detect_column_role(series),
            semantic_role=self.detect_semantic_role(stats),
            profile=self.build_statistical_profile(stats),
            is_target=is_target,
            domain_profile=self.domain_profiler.build_for_categorical(domain_tags),
            task_type=self.resolve_task_type(stats, task_type),
            categorical_profile=self.build_categorical_profile(stats, domain_tags),
        )

    def _infer_task_type(self, stats: CategoricalColumnStats) -> Decision[TaskType]:
        return self.task_policy.resolve(CategoricalRuleContext(stats=stats))


class AnalyzerRegistry:
    def __init__(
        self,
        analyzers: tuple[ColumnAnalyzer[Any], ...] | None = None,
        config: AnalyzerConfig | None = None,
        domain_policy: DomainPolicy | None = None,
        domain_profiler: DomainProfiler | None = None,
    ) -> None:
        self.analyzers = analyzers or ColumnAnalyzer.create_registered(
            config=config,
            domain_policy=domain_policy,
            domain_profiler=domain_profiler,
        )

    def get_analyzer(self, series: pd.Series) -> ColumnAnalyzer[Any]:
        matches = [
            analyzer for analyzer in self.analyzers if analyzer.can_analyze(series)
        ]

        if not matches:
            raise ValueError(f"No analyzer registered for column '{series.name}'")
        
        if len(matches) > 1:
            analyzer_names = ", ".join(type(analyzer).__name__ for analyzer in matches)
            raise ValueError(
                f"Ambiguous analyzers for column '{series.name}': {analyzer_names}"
            )
        
        return matches[0]
