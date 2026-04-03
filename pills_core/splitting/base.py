from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from pills_core._enums import ValidationStrategy
from pills_core.splitting.spec import FoldIndices, SplitRequest, SplitSpec


def _validate_request(request: SplitRequest) -> None:
    if not isinstance(request, SplitRequest):
        raise TypeError("request must be a SplitRequest.")
    if request.n_samples < 2:
        raise ValueError("Splitting requires at least two samples.")


def _validate_folds(request: SplitRequest, folds: tuple[FoldIndices, ...]) -> None:
    if len(folds) == 0:
        raise ValueError("Splitter must produce at least one fold.")

    for fold in folds:
        if not isinstance(fold, FoldIndices):
            raise TypeError("Splitter must produce FoldIndices instances.")
        fold.validate_bounds(request.n_samples)
        if fold.train_size + fold.val_size > request.n_samples:
            raise ValueError(
                "Fold indices cannot reference more samples than exist in the request."
            )


class BaseSplitter(ABC):
    strategy: ClassVar[ValidationStrategy]

    def build(self, request: SplitRequest) -> SplitSpec:
        _validate_request(request)

        folds = tuple(self._build_folds(request))
        _validate_folds(request, folds)

        return SplitSpec.from_folds(
            strategy=self._get_strategy(),
            random_state=self._get_random_state(),
            n_samples=request.n_samples,
            folds=folds,
        )

    def build_spec(self, df: pd.DataFrame, y: pd.Series) -> SplitSpec:
        return self.build(SplitRequest(frame=df, target=y))

    def _get_strategy(self) -> ValidationStrategy:
        try:
            strategy = self.strategy
        except AttributeError as exc:
            raise TypeError(
                f"{type(self).__name__} must define a strategy attribute."
            ) from exc

        if not isinstance(strategy, ValidationStrategy):
            raise TypeError(
                f"{type(self).__name__}.strategy must be a ValidationStrategy."
            )

        return strategy

    @abstractmethod
    def _build_folds(self, request: SplitRequest) -> tuple[FoldIndices, ...]: ...

    @abstractmethod
    def _get_random_state(self) -> int: ...
