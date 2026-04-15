from __future__ import annotations

from dataclasses import dataclass

from pills_core.strategies.base import ColumnMeta, StrategyEmbedding
from pills_core.types.stats import BaseColumnStats


@dataclass(frozen=True)
class ColumnContext:
    name: str
    stats: BaseColumnStats
    meta: ColumnMeta
    embedding: StrategyEmbedding
