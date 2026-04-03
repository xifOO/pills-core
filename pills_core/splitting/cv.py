from typing import ClassVar

from sklearn.model_selection import KFold, StratifiedKFold

from pills_core._enums import ValidationStrategy
from pills_core.config import TrainingConfig
from pills_core.splitting.base import BaseSplitter
from pills_core.splitting.spec import FoldIndices, SplitRequest


class CVSplitter(BaseSplitter):
    strategy: ClassVar[ValidationStrategy] = ValidationStrategy.CV

    def __init__(self, config: TrainingConfig, stratify: bool = True) -> None:
        self.config = config
        self.stratify = stratify

    def _build_folds(self, request: SplitRequest) -> tuple[FoldIndices, ...]:
        if self.stratify:
            skf = StratifiedKFold(
                n_splits=self.config.folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
        else:
            skf = KFold(
                n_splits=self.config.folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
        return tuple(
            FoldIndices(train=train_idx, val=val_idx)
            for train_idx, val_idx in skf.split(request.frame, request.target)
        )

    def _get_random_state(self) -> int:
        return int(self.config.random_state)
