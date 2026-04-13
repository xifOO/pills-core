from typing import Callable, TypeVar

ClsT = TypeVar("ClsT", bound=type)


def register_required_stats(*stats: str) -> Callable[[ClsT], ClsT]:
    """
    Decorator for register required stats for strategy.
    Example:
        @register_required_stats("median", "mean")
        class MedianImputation:
    """

    def decorator(cls: ClsT) -> ClsT:
        parent_stats = set()
        for base in cls.__mro__:
            if hasattr(base, "required_stats"):
                parent_stats.update(base.required_stats)
        cls.required_stats = parent_stats.union(stats)
        return cls

    return decorator
