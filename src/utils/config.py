"""Pydantic settings for the compression layer project."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    tinker_api_key: str = Field(default="", alias="TINKER_API_KEY")

    # Cost tracking
    cost_limit_daily_usd: float = Field(default=50.0, alias="COST_LIMIT_DAILY_USD")
    cost_warn_threshold_usd: float = Field(default=30.0, alias="COST_WARN_THRESHOLD_USD")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    # Paths (relative to project root)
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def seed_dir(self) -> Path:
        return self.data_dir / "seed"

    @property
    def validated_dir(self) -> Path:
        return self.data_dir / "validated"

    @property
    def costs_log(self) -> Path:
        return self.data_dir / "costs.log"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def adapters_dir(self) -> Path:
        return self.models_dir / "adapters"


class GenerationConfig(BaseSettings):
    """Settings specific to compression pair generation."""

    model_config = SettingsConfigDict(extra="ignore")

    primary_model: str = "claude-sonnet-4-20250514"
    fallback_model: str = "gpt-4o-mini"
    temperature: float = 0.4
    max_retries: int = 3
    concurrency: int = 10


class ValidationConfig(BaseSettings):
    """Settings specific to cross-model validation."""

    model_config = SettingsConfigDict(extra="ignore")

    models: list[str] = [
        "claude-sonnet-4-20250514",
        "gpt-4o-mini",
        "gemini-2.0-flash",
    ]
    equivalence_threshold: float = 0.85
    tasks: list[str] = ["qa", "reasoning"]
    concurrency: int = 10


class TrainingConfig(BaseSettings):
    """Settings specific to model training."""

    model_config = SettingsConfigDict(extra="ignore")

    # Local MLX training
    local_model: str = "mlx-community/Qwen3-4B-Instruct-4bit"
    local_lora_rank: int = 8
    local_lora_alpha: int = 16
    local_iters: int = 500
    local_batch_size: int = 2
    local_lr: float = 1e-4

    # Cloud Tinker training
    cloud_model: str = "Qwen/Qwen3-8B"
    cloud_lora_rank: int = 64
    cloud_lora_alpha: int = 128
    cloud_epochs: int = 3
    cloud_batch_size: int = 4
    cloud_lr: float = 2e-4


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
