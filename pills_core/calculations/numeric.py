import pandas as pd

from pills_core.types.stats import NumericalColumnStats


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
