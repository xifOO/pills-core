from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class DecisionScore:
    """
    Priority assembled from components.
    base      — how good the strategy is in general
    condition — how well the current stats fit this strategy
    penalty   — risk penalty (outliers, skewness, small sample)

    Scale guide:
        base:      300 (strong general strategy)
                   250 (good but situational)
                   150 (transformation — changes shape, use carefully)
        condition: 150 (perfect match)
                   100 (strong match)
                    75 (moderate match)
                    25 (minor bonus)
        penalty:   150 (critical mismatch — should rarely win)
                   100 (significant risk)
                    50 (minor concern)
    """

    base: int
    condition: int
    penalty: int = 0

    @property
    def total(self) -> int:
        return self.base + self.condition - self.penalty
