from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from pills_core._enums import ValidationStrategy


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_positions(values: np.ndarray, *, name: str) -> np.ndarray:
    positions = np.asarray(values)

    if positions.ndim != 1:
        raise ValueError(f"{name} indices must be one-dimensional.")
    if positions.size == 0:
        raise ValueError(f"{name} indices must not be empty.")
    if not np.issubdtype(positions.dtype, np.integer):
        raise TypeError(f"{name} indices must be integer positions.")

    normalized = positions.astype(np.intp, copy=False)
    if np.unique(normalized).size != normalized.size:
        raise ValueError(f"{name} indices must be unique within a fold.")

    return normalized


@dataclass(frozen=True, slots=True)
class SplitRequest:
    frame: pd.DataFrame
    target: pd.Series

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame.")
        if not isinstance(self.target, pd.Series):
            raise TypeError("target must be a pandas Series.")
        if len(self.frame.index) == 0:
            raise ValueError("frame must contain at least one row.")
        if len(self.frame) != len(self.target):
            raise ValueError(
                "frame and target must contain the same number of samples."
            )
        if not self.frame.index.equals(self.target.index):
            raise ValueError("frame and target must share the same index.")

    @property
    def n_samples(self) -> int:
        return int(len(self.frame))

    @property
    def positions(self) -> np.ndarray:
        return np.arange(self.n_samples, dtype=np.intp)


@dataclass(frozen=True, slots=True)
class FoldIndices:
    train: np.ndarray
    val: np.ndarray

    def __post_init__(self) -> None:
        train = _normalize_positions(self.train, name="train")
        val = _normalize_positions(self.val, name="val")

        overlap = np.intersect1d(train, val, assume_unique=True)
        if overlap.size > 0:
            raise ValueError("train and val indices must be disjoint within a fold.")

        object.__setattr__(self, "train", train)
        object.__setattr__(self, "val", val)

    @property
    def train_size(self) -> int:
        return int(self.train.size)

    @property
    def val_size(self) -> int:
        return int(self.val.size)

    def validate_bounds(self, n_samples: int) -> None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        max_index = n_samples - 1
        if self.train.min() < 0 or self.val.min() < 0:
            raise ValueError("Fold indices must be non-negative.")
        if self.train.max() > max_index or self.val.max() > max_index:
            raise ValueError(f"Fold indices must be in the range [0, {max_index}].")

    def __repr__(self) -> str:
        return (
            f"FoldIndices(train={self.train_size} samples, val={self.val_size} samples)"
        )


@dataclass(frozen=True, slots=True)
class SplitDiagnostics:
    n_samples: int
    n_folds: int
    train_sizes: tuple[int, ...]
    val_sizes: tuple[int, ...]
    unique_train_coverage: int
    unique_val_coverage: int
    train_coverage_ratio: float
    val_coverage_ratio: float
    has_validation_overlap: bool
    is_exhaustive_validation: bool

    @classmethod
    def from_folds(
        cls,
        n_samples: int,
        folds: tuple[FoldIndices, ...],
    ) -> "SplitDiagnostics":
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if len(folds) == 0:
            raise ValueError("folds must not be empty.")

        train_sizes = tuple(fold.train_size for fold in folds)
        val_sizes = tuple(fold.val_size for fold in folds)

        train_concat = np.concatenate([fold.train for fold in folds])
        val_concat = np.concatenate([fold.val for fold in folds])

        unique_train_coverage = int(np.unique(train_concat).size)
        unique_val_coverage = int(np.unique(val_concat).size)

        train_coverage_ratio = unique_train_coverage / n_samples
        val_coverage_ratio = unique_val_coverage / n_samples
        has_validation_overlap = unique_val_coverage != int(val_concat.size)
        is_exhaustive_validation = unique_val_coverage == n_samples

        return cls(
            n_samples=n_samples,
            n_folds=len(folds),
            train_sizes=train_sizes,
            val_sizes=val_sizes,
            unique_train_coverage=unique_train_coverage,
            unique_val_coverage=unique_val_coverage,
            train_coverage_ratio=train_coverage_ratio,
            val_coverage_ratio=val_coverage_ratio,
            has_validation_overlap=has_validation_overlap,
            is_exhaustive_validation=is_exhaustive_validation,
        )


@dataclass(frozen=True, slots=True)
class SplitSpec:
    strategy: ValidationStrategy
    random_state: int
    n_samples: int
    folds: tuple[FoldIndices, ...]
    created_at: str = field(default_factory=_utc_now_iso)
    diagnostics: SplitDiagnostics = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, ValidationStrategy):
            raise TypeError("strategy must be a ValidationStrategy.")
        if not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer.")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if len(self.folds) == 0:
            raise ValueError("folds must not be empty.")

        normalized_folds = tuple(self.folds)
        for fold in normalized_folds:
            if not isinstance(fold, FoldIndices):
                raise TypeError("folds must contain FoldIndices objects.")
            fold.validate_bounds(self.n_samples)

        object.__setattr__(self, "folds", normalized_folds)
        object.__setattr__(
            self,
            "diagnostics",
            SplitDiagnostics.from_folds(
                n_samples=self.n_samples,
                folds=normalized_folds,
            ),
        )

    @classmethod
    def from_folds(
        cls,
        strategy: ValidationStrategy,
        random_state: int,
        n_samples: int,
        folds: tuple[FoldIndices, ...],
    ) -> "SplitSpec":
        return cls(
            strategy=strategy,
            random_state=random_state,
            n_samples=n_samples,
            folds=folds,
        )

    def get_fold(self, fold_id: int) -> FoldIndices:
        if fold_id < 0 or fold_id >= len(self.folds):
            raise IndexError(
                f"fold_id={fold_id} out of range, spec has {len(self.folds)} folds."
            )
        return self.folds[fold_id]

    @property
    def n_folds(self) -> int:
        return len(self.folds)
