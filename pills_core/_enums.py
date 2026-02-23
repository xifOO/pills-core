from enum import StrEnum, IntEnum


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
    ID = "id"


class TransformPhase(IntEnum):
    IMPUTATION = 1
    OUTLIER = 2
    SCALING = 3