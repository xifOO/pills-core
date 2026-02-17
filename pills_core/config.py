from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pills_core._enums import TaskType


class HardwareConfig(BaseSettings):
    use_gpu: bool = Field(default=False, description="Whether to use GPU")
    cpu_limit: int = Field(
        default=-1, description="Number of CPU cores (-1 = all available)"
    )
    max_memory_mb: int = Field(default=4096, description="RAM limit for training")


class DataProcessingConfig(BaseSettings):
    drop_uninformative: bool = Field(
        default=True, description="Drop columns with a single unique value"
    )
    handle_missing: bool = Field(
        default=True, description="Automatically fill missing values"
    )
    max_unique_categories: int = Field(
        default=100, description="Max unique values limit for a categorical column"
    )
    date_features: bool = Field(
        default=True,
        description="Decompose dates into components (day, month, hour, ...)",
    )


class TrainingConfig(BaseSettings):
    timeout: int = Field(default=3600, description="Timeout in seconds")
    random_state: int = 42
    fast_mode: bool = Field(
        default=True,
        description="If True — fast models, if False — heavy models (CatBoost)",
    )
    validation_strategy: str = Field(
        default="cv",
        description="Validation strategy: 'cv' (cross-validation) or 'holdout'",
    )
    folds: int = Field(default=5, description="Number of folds for cross-validation")


class PillConfig(BaseSettings):
    project_name: str = Field(
        default="my_pills_project", description="Project name for logs and saving"
    )
    task_type: TaskType = Field(default=TaskType.AUTO)
    target: Optional[str] = Field(
        default=None, description="Target column name (if None — use the last column)"
    )
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    data: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    model_config = SettingsConfigDict(
        title="Pills Main Configuration",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        json_schema_extra={
            "example": {
                "project_name": "credit_scoring",
                "task_type": "binary",
                "training": {"timeout": 600},
            }
        },
    )


config = PillConfig()  # type: ignore[call-arg]
