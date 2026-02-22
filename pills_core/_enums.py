from enum import Enum


class TaskType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    AUTO = "auto"


class ColumnRole(Enum):
    TARGET = "target"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    DROP = "drop"
    ID = "id"
