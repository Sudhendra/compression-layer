"""API cost tracking and logging utilities."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .config import get_settings


class APICallRecord(BaseModel):
    """Record of a single API call for cost tracking."""

    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    operation: str  # e.g., "generation", "validation", "embedding"


# Pricing per 1M tokens (input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Claude
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    # Gemini
    "gemini-2.0-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculate the cost of an API call.

    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    input_rate, output_rate = MODEL_PRICING.get(model, (1.0, 1.0))
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    return input_cost + output_cost


class CostTracker:
    """
    Track API costs and enforce spending limits.

    Logs all API calls to a JSONL file and provides daily spend summaries.
    """

    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path or get_settings().costs_log
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._daily_limit = get_settings().cost_limit_daily_usd
        self._warn_threshold = get_settings().cost_warn_threshold_usd

    def log_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "unknown",
    ) -> float:
        """
        Log an API call and return its cost.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation: Type of operation (e.g., "generation", "validation")

        Returns:
            Cost of this call in USD
        """
        cost = calculate_cost(model, input_tokens, output_tokens)

        record = APICallRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            operation=operation,
        )

        with open(self.log_path, "a") as f:
            f.write(record.model_dump_json() + "\n")

        return cost

    def get_daily_spend(self, target_date: date | None = None) -> float:
        """Get total spend for a specific date (defaults to today)."""
        if target_date is None:
            target_date = date.today()

        total = 0.0
        if not self.log_path.exists():
            return total

        with open(self.log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_date = datetime.fromisoformat(record["timestamp"]).date()
                if record_date == target_date:
                    total += record["cost_usd"]

        return total

    def get_total_spend(self) -> float:
        """Get total spend across all time."""
        total = 0.0
        if not self.log_path.exists():
            return total

        with open(self.log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                total += record["cost_usd"]

        return total

    def check_limit(self) -> Literal["ok", "warn", "exceeded"]:
        """
        Check if we're within spending limits.

        Returns:
            "ok" - Under warning threshold
            "warn" - Over warning threshold but under limit
            "exceeded" - Over daily limit
        """
        daily_spend = self.get_daily_spend()

        if daily_spend >= self._daily_limit:
            return "exceeded"
        elif daily_spend >= self._warn_threshold:
            return "warn"
        return "ok"

    def get_spend_summary(self) -> dict[str, float]:
        """Get spend breakdown by model."""
        summary: dict[str, float] = {}

        if not self.log_path.exists():
            return summary

        with open(self.log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                model = record["model"]
                summary[model] = summary.get(model, 0.0) + record["cost_usd"]

        return summary


# Global tracker instance
_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker
