from enum import Enum


class TaskType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    AUTO = "auto"


class ColumnRole(Enum):
    TARGET = "target"
    NUMERIC = "numeric"
    CATEGORY = "category"
    DATETIME = "datetime"
    DROP = "drop"
    ID = "id"
