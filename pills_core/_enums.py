from enum import Enum, StrEnum, IntEnum, auto


class TaskType(StrEnum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    AUTO = "auto"


class ColumnRole(StrEnum):
    TARGET = "target"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    DROP = "drop"


class SemanticRole(Enum):
    CONTINUOUS = auto()
    COUNT = auto()
    ORDINAL = auto()
    NUMERIC_NOMINAL = auto()
    BINARY = auto()
    ID_LIKE = auto()


class TransformPhase(IntEnum):
    IMPUTATION = 1
    OUTLIER = 2
    SCALING = 3


class FamilyRole(StrEnum):
    CENTRAL_TENDENCY = auto()
    CONSTANT = auto()
    BOUNDARY = auto()
    STATISTICAL = auto()
    ROBUST = auto()
    PERCENTILE = auto()
    LINEAR_SCALING = auto()
    SKEW_TRANSFORM = auto()
