from enum import Enum, StrEnum, auto


class TaskType(StrEnum):
    BINARY = auto()
    MULTICLASS = auto()
    REGRESSION = auto()
    TIME_SERIES = auto()
    AUTO = auto()


class ColumnRole(StrEnum):
    TARGET = auto()
    NUMERICAL = auto()
    CATEGORICAL = auto()
    DATETIME = auto()
    DROP = auto()


class SemanticRole(Enum):
    CONTINUOUS = auto()
    COUNT = auto()
    ORDINAL = auto()
    NUMERIC_NOMINAL = auto()
    BINARY = auto()
    ID_LIKE = auto()


class TransformPhase(StrEnum):
    IMPUTATION = auto()
    OUTLIER = auto()
    SCALING = auto()


class FamilyRole(StrEnum):
    CENTRAL_TENDENCY = auto()
    CONSTANT = auto()
    BOUNDARY = auto()
    STATISTICAL = auto()
    ROBUST = auto()
    PERCENTILE = auto()
    LINEAR_SCALING = auto()
    SKEW_TRANSFORM = auto()


class DriftSeverity(StrEnum):
    STABLE = auto()
    MODERATE = auto()
    CRITICAL = auto()

    @classmethod
    def from_psi(cls, psi: float) -> "DriftSeverity":
        if psi < 0.1:
            return cls.STABLE
        elif psi < 0.2:
            return cls.MODERATE
        return cls.CRITICAL
