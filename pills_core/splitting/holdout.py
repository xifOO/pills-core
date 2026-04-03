from typing import ClassVar

from sklearn.model_selection import train_test_split

from pills_core._enums import ValidationStrategy
from pills_core.splitting.base import BaseSplitter
from pills_core.splitting.spec import FoldIndices, SplitRequest


class HoldoutSplitter(BaseSplitter):
    strategy: ClassVar[ValidationStrategy] = ValidationStrategy.HOLDOUT

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")

        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def _build_folds(self, request: SplitRequest) -> tuple[FoldIndices, ...]:
        train_idx, val_idx = train_test_split(
            request.positions,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=request.target if self.stratify else None,
        )

        return (FoldIndices(train=train_idx, val=val_idx),)

    def _get_random_state(self) -> int:
        return int(self.random_state)
