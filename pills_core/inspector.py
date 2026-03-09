import pandas as pd

from pills_core.analyzers import NumericalColumnAnalyzer
from pills_core.ingestion.csv import CSVDataSource, CSVOptions
from pills_core.strategies.base import ColumnMeta
from pills_core.strategies.numeric._registry import (
    build_imputation_registry,
    build_outliers_registry,
    build_scaling_registry,
)
from pills_core.types.stats import NumericalColumnStats

analyzer = NumericalColumnAnalyzer()


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
    series: pd.Series, stats: NumericalColumnStats, is_target: bool
) -> ColumnMeta:
    role = analyzer.detect_column_role(series)
    semantic_role = analyzer.detect_semantic_role(stats)
    profile = analyzer.build_statistical_profile(stats)
    return ColumnMeta(
        role=role,
        semantic_role=semantic_role,
        profile=profile,
        is_target=is_target,
    )


def process_column(
    series: pd.Series,
    stats: NumericalColumnStats,
    is_target: bool = False,
    verbose: bool = True,
) -> pd.Series:
    imputation_registry = build_imputation_registry()
    outliers_registry = build_outliers_registry()
    scaling_registry = build_scaling_registry()

    meta = build_meta(series, stats, is_target)
    column_embedding = analyzer.build_column_embedding(stats, meta)  # <--

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Column: '{series.name}' | is_target={is_target}")
        print(f"  semantic_role={meta.semantic_role.name} | profile={meta.profile}")
        print(
            f"  missing={stats.missing_ratio:.1%} | skewness={stats.skewness:.2f} | outlier_ratio={stats.outlier_ratio:.1%}"
        )
        print(
            f"  mean={stats.mean:.2f} | std={stats.std:.2f} | min={stats.min:.2f} | max={stats.max:.2f}"
        )
        print(
            f"  unique_ratio={stats.unique_ratio:.2f} | zero_ratio={stats.zero_ratio:.2f} | cv={stats.cv:.3f}"
        )
        print(f"  column_embedding={column_embedding}")

    result = series.copy()

    # imputation
    imputation_explain = imputation_registry.explain(meta, column_embedding, stats)
    result = imputation_registry.apply(result, meta, column_embedding, stats)
    if verbose:
        print("\n[IMPUTATION]")
        for line in imputation_explain:
            print(f"  {line}")

    stats = compute_stats(result)
    meta = build_meta(result, stats, is_target)
    column_embedding = analyzer.build_column_embedding(
        stats, meta
    )  # пересчитываем после imputation

    # outliers
    outliers_explain = outliers_registry.explain(meta, column_embedding, stats)
    result = outliers_registry.apply(result, meta, column_embedding, stats)
    if verbose:
        print("\n[OUTLIERS]")
        if outliers_explain:
            for line in outliers_explain:
                print(f"  {line}")
        else:
            print("  no strategies applied")

    stats = compute_stats(result)
    column_embedding = analyzer.build_column_embedding(
        stats, meta
    )  # пересчитываем после outliers

    # scaling
    scaling_explain = scaling_registry.explain(meta, column_embedding, stats)
    result = scaling_registry.apply(result, meta, column_embedding, stats)
    if verbose:
        print("\n[SCALING]")
        if scaling_explain:
            for line in scaling_explain:
                print(f"  {line}")
        else:
            print("  no strategies applied")

    return result


def main() -> None:
    options = CSVOptions(path="clean.csv")
    datasource = CSVDataSource(options)
    df = datasource.load()
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
        processed = process_column(df[col], stats, is_target=False)
        processed_parts.append(processed)

    if pd.api.types.is_numeric_dtype(df[target_col]):
        stats = compute_stats(df[target_col])
        processed_target = process_column(df[target_col], stats, is_target=True)
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
