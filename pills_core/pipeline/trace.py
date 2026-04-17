from dataclasses import dataclass
from typing import Tuple

from pills_core._enums import TransformPhase
from pills_core.strategies.base import SingleStrategy


@dataclass(frozen=True, slots=True)
class PhaseTrace:
    phase: TransformPhase
    candidates: Tuple[Tuple[SingleStrategy, float], ...]
    winner: SingleStrategy


