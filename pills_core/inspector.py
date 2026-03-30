from typing import Any, Mapping

import pandas as pd

from pills_core._enums import TaskType, TransformPhase
from pills_core.analyzers import (
    AnalyzerBuilder,
    AnalyzerConfig,
    DomainConfig,
    DomainRuleConfig,
    NumericalColumnAnalyzer,
)
from pills_core.ingestion.csv import CSVDataSource, CSVOptions
from pills_core.rules import DomainTags
from pills_core.strategies.config import NumericStrategiesConfig
from pills_core.strategies.numeric._registry import (
    build_imputation_registry,
    build_outliers_registry,
    build_scaling_registry,
)
from pills_core.strategies.resolver import resolve_phase_order
from pills_core.types.stats import (
    CategoricalThresholds,
    NumericalColumnStats,
    NumericalThresholds,
)


def _build_domain_config_from_rules(
    rules_payload: list[Mapping[str, Any]],
) -> DomainConfig:
    return DomainConfig(
        rules=tuple(build_domain_rule_config(rule) for rule in rules_payload)
    )


def _build_domain_config_from_keywords(
    keywords_payload: Mapping[str, Any],
) -> DomainConfig:
    rules: list[DomainRuleConfig] = []
    for name, keywords in keywords_payload.items():
        tags = DomainTags.from_group_name(str(name))
        rules.append(
            DomainRuleConfig(
                name=str(name),
                keywords=tuple(str(k) for k in keywords),
                tags=tags,
            )
        )
    return DomainConfig(rules=tuple(rules))


def build_domain_config(payload: Mapping[str, Any]) -> DomainConfig:
    if not payload:
        return DomainConfig()

    rules_payload = payload.get("rules")
    if rules_payload is not None:
        return _build_domain_config_from_rules(rules_payload)

    keywords_payload = payload.get("keywords", payload)
    if isinstance(keywords_payload, Mapping):
        return _build_domain_config_from_keywords(keywords_payload)

    return DomainConfig()


def build_domain_rule_config(payload: Mapping[str, Any]) -> DomainRuleConfig:
    rule_name = str(payload["name"])
    tags_payload = payload.get("tags")

    if tags_payload is None:
        tags = DomainTags.from_group_name(rule_name)
    elif not isinstance(tags_payload, Mapping):
        raise TypeError("domain.rules[].tags must be a mapping.")
    else:
        tags = DomainTags(
            is_ratio=bool(tags_payload.get("is_ratio", False)),
            is_monetary=bool(tags_payload.get("is_monetary", False)),
            is_rate=bool(tags_payload.get("is_rate", False)),
            is_score=bool(tags_payload.get("is_score", False)),
            is_count=bool(tags_payload.get("is_count", False)),
        )

    return DomainRuleConfig(
        name=rule_name,
        keywords=tuple(str(k) for k in payload.get("keywords", ())),
        tags=tags,
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
            DomainRuleConfig(
                name="count",
                keywords=(
                    "acc",
                    "inq",
                    "delinq",
                    "rec",
                    "tl",
                    "mths",
                    "mo_sin",
                    "num_",
                    "open_",
                    "pub_rec",
                    "collections",
                    "bankruptcies",
                    "liens",
                    "term",
                ),
                tags=DomainTags(is_count=True),
            ),
        )
    )


analyzer_registry = AnalyzerBuilder(
    config=AnalyzerConfig(
        numerical=NumericalThresholds(),
        categorical=CategoricalThresholds(),
        domain=default_domain_config(),
    )
).build_registry()


def compute_stats(series: pd.Series) -> NumericalColumnStats:
    clean = series.dropna()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)

    count = len(clean)
    n_unique = clean.nunique()
    std = clean.std()
    mean = clean.mean()

    diffs = clean.sort_index().diff().dropna()
    monotonic_ratio = float((diffs > 0).sum() / len(diffs)) if len(diffs) > 0 else 0.0

    return NumericalColumnStats(
        max=clean.max(),
        min=clean.min(),
        mean=mean,
        median=clean.median(),
        mode=clean.mode()[0] if not clean.mode().empty else clean.median(),
        std=std,
        count=count,
        variance=clean.var(),
        skewness=pd.to_numeric(clean.skew()),
        kurtosis=pd.to_numeric(clean.kurt()),
        range=clean.max() - clean.min(),
        n_unique=n_unique,
        missing_ratio=float(series.isna().mean()),
        outlier_ratio=float(outlier_mask.mean()),
        q1=q1,
        q3=q3,
        p05=clean.quantile(0.05),
        p95=clean.quantile(0.95),
        is_integer_valued=bool((clean == clean.round()).all()),
        monotonic_ratio=monotonic_ratio,
        cv=float(abs(std / mean)) if mean != 0 else 0.0,
        unique_ratio=float(n_unique / count) if count > 0 else 0.0,
        zero_ratio=float((clean == 0).mean()),
    )


def build_meta(
    series: pd.Series,
    stats: NumericalColumnStats,
    is_target: bool,
    analyzer: NumericalColumnAnalyzer,
):
    return analyzer.build_meta(
        series=series,
        stats=stats,
        is_target=is_target,
        task_type=TaskType.AUTO,
    )


def process_column(
    series: pd.Series,
    stats: NumericalColumnStats,
    is_target: bool = False,
    verbose: bool = True,
    strategy_config: NumericStrategiesConfig | None = None,
) -> pd.Series:
    analyzer = analyzer_registry.get_analyzer(series)
    if not isinstance(analyzer, NumericalColumnAnalyzer):
        raise TypeError(f"Expected numerical analyzer for column '{series.name}'.")

    if strategy_config is None:
        strategy_config = NumericStrategiesConfig()

    imputation_registry = build_imputation_registry(strategy_config.imputation)
    outliers_registry = build_outliers_registry(strategy_config.outlier)
    scaling_registry = build_scaling_registry(strategy_config.scaling)

    meta = build_meta(series, stats, is_target, analyzer)
    column_embedding = analyzer.build_column_embedding(stats, meta)

    imputation_candidates = imputation_registry.resolve(meta, column_embedding, stats)
    outlier_candidates = outliers_registry.resolve(meta, column_embedding, stats)
    scaling_candidates = scaling_registry.resolve(meta, column_embedding, stats)

    best_imputation = imputation_candidates[0][0] if imputation_candidates else None
    best_outlier = outlier_candidates[0][0] if outlier_candidates else None
    best_scaling = scaling_candidates[0][0] if scaling_candidates else None

    if best_imputation and best_outlier and best_scaling:
        phase_order = resolve_phase_order(best_imputation, best_outlier, best_scaling)
    else:
        phase_order = [
            TransformPhase.IMPUTATION,
            TransformPhase.OUTLIER,
            TransformPhase.SCALING,
        ]

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Column: '{series.name}' | is_target={is_target}")
        print(
            f"  semantic_role={meta.semantic_role.name} | profile={meta.profile} | task_type={meta.task_type}"
        )
        print(
            f"  missing={stats.missing_ratio:.1%} | skewness={stats.skewness:.2f} | outlier_ratio={stats.outlier_ratio:.1%}"
        )
        print(f"  column_embedding={column_embedding}")
        print(f"\n  resolved order: {' → '.join(p.value.upper() for p in phase_order)}")
        if best_imputation:
            print(f"  best_imputation : {best_imputation.name}")
        if best_outlier:
            print(f"  best_outlier    : {best_outlier.name}")
        if best_scaling:
            print(f"  best_scaling    : {best_scaling.name}")

    phase_registry_map = {
        TransformPhase.IMPUTATION: imputation_registry,
        TransformPhase.OUTLIER: outliers_registry,
        TransformPhase.SCALING: scaling_registry,
    }
    explain_map = {
        TransformPhase.IMPUTATION: imputation_registry.explain(
            meta, column_embedding, stats
        ),
        TransformPhase.OUTLIER: outliers_registry.explain(
            meta, column_embedding, stats
        ),
        TransformPhase.SCALING: scaling_registry.explain(meta, column_embedding, stats),
    }

    result = series.copy()

    for phase in phase_order:
        phase_registry = phase_registry_map[phase]

        if verbose:
            print(f"\n[{phase.value.upper()}]")
            for line in explain_map[phase]:
                print(f"  {line}")

        result = phase_registry.apply(result, meta, column_embedding, stats)

        stats = compute_stats(result)
        meta = build_meta(result, stats, is_target, analyzer)
        column_embedding = analyzer.build_column_embedding(stats, meta)

    return result


def main() -> None:
    options = CSVOptions(path="clean.csv")
    datasource = CSVDataSource(options)
    df = datasource.load()
    strategy_config = NumericStrategiesConfig()
    print("=== RAW DATA ===")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isna().sum()}")

    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    processed_parts = []

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"\nSkipping '{col}' — not numeric")
            processed_parts.append(df[col])
            continue

        stats = compute_stats(df[col])
        processed = process_column(
            df[col], stats, is_target=False, strategy_config=strategy_config
        )
        processed_parts.append(processed)

    if pd.api.types.is_numeric_dtype(df[target_col]):
        stats = compute_stats(df[target_col])
        processed_target = process_column(
            df[target_col], stats, is_target=True, strategy_config=strategy_config
        )
        processed_parts.append(processed_target)
    else:
        processed_parts.append(df[target_col])

    result_df = pd.concat(processed_parts, axis=1)

    print("\n\n=== PROCESSED DATA ===")
    print(result_df.head(10))
    print(f"\nMissing after processing: {result_df.isna().sum().sum()}")

    result_df.to_csv("numeric_processed.csv", index=False)
    print("\nSaved to numeric_processed.csv")


if __name__ == "__main__":
    main()
