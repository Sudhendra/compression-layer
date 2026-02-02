#!/usr/bin/env python3
"""
MLX LoRA Training Log Parser and Visualizer with Run Management

Features:
- Automatically finds latest training run
- Detects training failures
- Supports specific run timestamps
- Organizes visualizations by timestamp
- Handles multiple training versions
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingMetrics:
    """Container for parsed training metrics"""

    iterations: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_iterations: list[int] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    iterations_per_sec: list[float] = field(default_factory=list)
    tokens_per_sec: list[float] = field(default_factory=list)
    trained_tokens: list[int] = field(default_factory=list)
    peak_memory: list[float] = field(default_factory=list)

    # Metadata
    total_iters: int | None = None
    trainable_params: str | None = None
    trainable_percentage: float | None = None

    # Run information
    timestamp: str | None = None
    log_path: Path | None = None


class TrainingRunManager:
    """Manage multiple training runs and find the latest/specific runs"""

    TIMESTAMP_FORMAT = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"

    def __init__(self, base_dir: Path = None):
        """Initialize run manager.

        Args:
            base_dir: Base directory for training runs (default: models/runs/mlx)
        """
        if base_dir is None:
            # scripts/ and models/ are siblings in the same folder
            script_dir = Path(__file__).parent  # This is scripts/
            project_root = script_dir.parent  # Go up to project root
            base_dir = project_root / "models" / "runs" / "mlx"

        self.base_dir = Path(base_dir)

    def get_all_runs(self) -> list[tuple[str, Path]]:
        """Get all training runs sorted by timestamp (newest first).

        Returns:
            List of (timestamp, run_dir) tuples
        """
        if not self.base_dir.exists():
            return []

        runs = []
        timestamp_pattern = re.compile(self.TIMESTAMP_FORMAT)

        for item in self.base_dir.iterdir():
            if item.is_dir() and timestamp_pattern.match(item.name):
                runs.append((item.name, item))

        # Sort by timestamp (newest first)
        runs.sort(reverse=True, key=lambda x: x[0])
        return runs

    def get_latest_run(self) -> tuple[str, Path] | None:
        """Get the most recent training run.

        Returns:
            (timestamp, run_dir) or None if no runs found
        """
        runs = self.get_all_runs()
        return runs[0] if runs else None

    def get_run(self, timestamp: str = None) -> tuple[str, Path] | None:
        """Get a specific run or the latest if timestamp not provided.

        Args:
            timestamp: Specific timestamp to get, or None for latest

        Returns:
            (timestamp, run_dir) or None if not found
        """
        if timestamp is None:
            return self.get_latest_run()

        # Validate timestamp format
        if not re.match(self.TIMESTAMP_FORMAT, timestamp):
            print(f"Error: Invalid timestamp format: {timestamp}")
            print("Expected format: YYYY-MM-DD_HH-MM-SS (e.g., 2026-02-01_01-28-37)")
            return None

        run_dir = self.base_dir / timestamp
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            return None

        return (timestamp, run_dir)

    def get_log_path(self, timestamp: str = None) -> Path | None:
        """Get the training log path for a specific run or latest.

        Args:
            timestamp: Specific timestamp or None for latest

        Returns:
            Path to train.log or None if not found
        """
        run = self.get_run(timestamp)
        if run is None:
            return None

        timestamp, run_dir = run
        log_path = run_dir / "train.log"

        if not log_path.exists():
            print(f"Error: Training log not found: {log_path}")
            return None

        return log_path

    def check_training_status(self, log_path: Path) -> tuple[bool, str | None]:
        """Check if training completed successfully or failed.

        Args:
            log_path: Path to the training log

        Returns:
            (success: bool, error_message: Optional[str])
        """
        if not log_path.exists():
            return False, "Log file not found"

        try:
            with open(log_path) as f:
                content = f.read()

            # Check for success indicators
            if "Saved final weights" in content:
                return True, None

            # Check for common error patterns
            error_patterns = [
                (r"Error:(.+)", "Error"),
                (r"Exception:(.+)", "Exception"),
                (r"Traceback \(most recent call last\)", "Exception/Traceback"),
                (r"CUDA out of memory", "Out of Memory"),
                (r"killed", "Process Killed"),
                (r"Failed", "Training Failed"),
            ]

            for pattern, error_type in error_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    if match.groups():
                        return False, f"{error_type}: {match.group(1).strip()}"
                    return False, error_type

            # No success marker and no errors - might be incomplete
            if "Starting training" in content:
                return False, "Training incomplete (no final weights saved)"

            return False, "Unknown status (log may be empty or corrupted)"

        except Exception as e:
            return False, f"Failed to read log: {str(e)}"

    def list_runs(self):
        """Print all available training runs."""
        runs = self.get_all_runs()

        if not runs:
            print("No training runs found.")
            return

        print(f"\nFound {len(runs)} training run(s):\n")
        print(f"{'Timestamp':<25} {'Directory':<50} {'Status'}")
        print("-" * 90)

        for timestamp, run_dir in runs:
            log_path = run_dir / "train.log"
            if log_path.exists():
                success, error = self.check_training_status(log_path)
                status = "✓ Complete" if success else f"✗ {error}"
            else:
                status = "✗ No log file"

            print(f"{timestamp:<25} {str(run_dir):<50} {status}")

        print()


class MLXLogParser:
    """Robust parser for MLX LoRA training logs"""

    # Regex patterns for different log formats
    PATTERNS = {
        "iter_info": re.compile(r"Iter (\d+)"),
        "train_loss": re.compile(r"Train loss ([\d.]+)"),
        "val_loss": re.compile(r"Val loss ([\d.]+)"),
        "learning_rate": re.compile(r"Learning Rate ([\d.e+-]+)"),
        "it_sec": re.compile(r"It/sec ([\d.]+)"),
        "tokens_sec": re.compile(r"Tokens/sec ([\d.]+)"),
        "trained_tokens": re.compile(r"Trained Tokens (\d+)"),
        "peak_mem": re.compile(r"Peak mem ([\d.]+) GB"),
        "total_iters": re.compile(r"iters: (\d+)"),
        "trainable_params": re.compile(
            r"Trainable parameters: ([\d.]+)% \(([\d.]+[KMB])/([\d.]+[KMB])\)"
        ),
    }

    @staticmethod
    def parse_size_string(size_str: str) -> float:
        """Convert size strings like '7.340M' to float"""
        multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}
        if size_str[-1] in multipliers:
            return float(size_str[:-1]) * multipliers[size_str[-1]]
        return float(size_str)

    def parse_log(self, log_content: str, log_path: Path = None) -> TrainingMetrics:
        """
        Parsing MLX training log and extract all metrics

        Args:
            log_content: String content of the log file
            log_path: Optional path to the log file for metadata

        Returns:
            TrainingMetrics object with parsed data
        """
        metrics = TrainingMetrics()

        # Extract timestamp from log path if provided
        if log_path:
            metrics.log_path = log_path
            # Extract timestamp from path like models/runs/mlx/2026-02-01_01-28-37/train.log
            timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", str(log_path))
            if timestamp_match:
                metrics.timestamp = timestamp_match.group(1)

        lines = log_content.strip().split("\n")

        # Parse metadata from header
        for line in lines[:10]:  # Check first 10 lines for metadata
            # Total iterations
            match = self.PATTERNS["total_iters"].search(line)
            if match:
                metrics.total_iters = int(match.group(1))

            # Trainable parameters
            match = self.PATTERNS["trainable_params"].search(line)
            if match:
                metrics.trainable_percentage = float(match.group(1))
                metrics.trainable_params = f"{match.group(2)}/{match.group(3)}"

        # Parse training iterations
        current_iter = None
        for line in lines:
            # Check if this is an iteration line
            iter_match = self.PATTERNS["iter_info"].search(line)
            if not iter_match:
                continue

            current_iter = int(iter_match.group(1))

            # Validation loss (special case - appears before train loss on same iter)
            val_match = self.PATTERNS["val_loss"].search(line)
            if val_match and "Train loss" not in line:
                metrics.val_losses.append(float(val_match.group(1)))
                metrics.val_iterations.append(current_iter)
                continue

            # Training loss
            train_match = self.PATTERNS["train_loss"].search(line)
            if train_match:
                metrics.iterations.append(current_iter)
                metrics.train_losses.append(float(train_match.group(1)))

                # Extract other metrics (only present with training loss)
                lr_match = self.PATTERNS["learning_rate"].search(line)
                if lr_match:
                    metrics.learning_rates.append(float(lr_match.group(1)))

                it_sec_match = self.PATTERNS["it_sec"].search(line)
                if it_sec_match:
                    metrics.iterations_per_sec.append(float(it_sec_match.group(1)))

                tokens_sec_match = self.PATTERNS["tokens_sec"].search(line)
                if tokens_sec_match:
                    metrics.tokens_per_sec.append(float(tokens_sec_match.group(1)))

                trained_tokens_match = self.PATTERNS["trained_tokens"].search(line)
                if trained_tokens_match:
                    metrics.trained_tokens.append(int(trained_tokens_match.group(1)))

                peak_mem_match = self.PATTERNS["peak_mem"].search(line)
                if peak_mem_match:
                    metrics.peak_memory.append(float(peak_mem_match.group(1)))

        return metrics


class MLXLogVisualizer:
    """Create comprehensive visualizations of training metrics"""

    def __init__(self, metrics: TrainingMetrics):
        self.metrics = metrics

    def create_individual_plots(self, output_dir: str = None, dpi: int = 150):
        """
        Create individual plot files for each metric

        Args:
            output_dir: Directory to save the figures (creates if doesn't exist)
            dpi: Resolution of the saved figures
        """
        if output_dir is None:
            print("Error: output_dir must be specified for individual plots")
            return

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # Plot 1: Loss Curves
        if len(self.metrics.train_losses) > 0 or len(self.metrics.val_losses) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            self._plot_losses(ax)
            self._add_metadata_title(fig, "Loss Curves")
            filename = output_path / "1_loss_curves.png"
            plt.savefig(filename, dpi=dpi, bbox_inches="tight")
            plt.close()
            saved_files.append(filename)
            print(f"  ✓ Saved: {filename.name}")

        # Plot 2: Learning Rate
        if len(self.metrics.learning_rates) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            self._plot_learning_rate(ax)
            self._add_metadata_title(fig, "Learning Rate Schedule")
            filename = output_path / "2_learning_rate.png"
            plt.savefig(filename, dpi=dpi, bbox_inches="tight")
            plt.close()
            saved_files.append(filename)
            print(f"  ✓ Saved: {filename.name}")

        # Plot 3: Training Speed
        if len(self.metrics.iterations_per_sec) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            self._plot_iterations_per_sec(ax)
            self._add_metadata_title(fig, "Training Speed (Iterations/sec)")
            filename = output_path / "3_training_speed.png"
            plt.savefig(filename, dpi=dpi, bbox_inches="tight")
            plt.close()
            saved_files.append(filename)
            print(f"  ✓ Saved: {filename.name}")

        # Plot 4: Throughput
        if len(self.metrics.tokens_per_sec) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            self._plot_tokens_per_sec(ax)
            self._add_metadata_title(fig, "Training Throughput (Tokens/sec)")
            filename = output_path / "4_throughput.png"
            plt.savefig(filename, dpi=dpi, bbox_inches="tight")
            plt.close()
            saved_files.append(filename)
            print(f"  ✓ Saved: {filename.name}")

        # Plot 5: Memory Usage
        if len(self.metrics.peak_memory) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            self._plot_memory(ax)
            self._add_metadata_title(fig, "Memory Usage")
            filename = output_path / "5_memory_usage.png"
            plt.savefig(filename, dpi=dpi, bbox_inches="tight")
            plt.close()
            saved_files.append(filename)
            print(f"  ✓ Saved: {filename.name}")

        # Plot 6: Token Progress
        if len(self.metrics.trained_tokens) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            self._plot_trained_tokens(ax)
            self._add_metadata_title(fig, "Token Progress")
            filename = output_path / "6_token_progress.png"
            plt.savefig(filename, dpi=dpi, bbox_inches="tight")
            plt.close()
            saved_files.append(filename)
            print(f"  ✓ Saved: {filename.name}")

        return saved_files

    def _add_metadata_title(self, fig, subtitle: str):
        """Add metadata to figure title"""
        title = f"MLX LoRA Training: {subtitle}"
        if self.metrics.timestamp:
            title += f"\nRun: {self.metrics.timestamp}"
        if self.metrics.trainable_params:
            title += f" | Trainable: {self.metrics.trainable_percentage:.3f}% ({self.metrics.trainable_params})"

        fig.suptitle(title, fontsize=13, fontweight="bold")

    def create_comprehensive_plot(self, output_path: str = None, dpi: int = 150):
        """
        Create a comprehensive visualization with multiple subplots (legacy - single file)

        Args:
            output_path: Path to save the figure (if None, displays instead)
            dpi: Resolution of the saved figure
        """
        # Determine layout based on available metrics
        has_performance = len(self.metrics.iterations_per_sec) > 0
        has_memory = len(self.metrics.peak_memory) > 0
        has_tokens = len(self.metrics.trained_tokens) > 0

        n_plots = 2  # Loss and LR always present
        if has_performance:
            n_plots += 1
        if has_memory:
            n_plots += 1
        if has_tokens:
            n_plots += 1

        # Create figure
        fig = plt.figure(figsize=(16, 3 * n_plots))
        gs = gridspec.GridSpec(n_plots, 2, figure=fig, hspace=0.3, wspace=0.3)

        plot_idx = 0

        # Plot 1: Training and Validation Loss
        ax1 = fig.add_subplot(gs[plot_idx, :])
        self._plot_losses(ax1)
        plot_idx += 1

        # Plot 2: Learning Rate
        if len(self.metrics.learning_rates) > 0:
            ax2 = fig.add_subplot(gs[plot_idx, :])
            self._plot_learning_rate(ax2)
            plot_idx += 1

        # Plot 3: Training Speed (if available)
        if has_performance:
            ax3 = fig.add_subplot(gs[plot_idx, 0])
            self._plot_iterations_per_sec(ax3)

            ax4 = fig.add_subplot(gs[plot_idx, 1])
            self._plot_tokens_per_sec(ax4)
            plot_idx += 1

        # Plot 4: Memory Usage (if available)
        if has_memory:
            ax5 = fig.add_subplot(gs[plot_idx, :])
            self._plot_memory(ax5)
            plot_idx += 1

        # Plot 5: Token Progress (if available)
        if has_tokens:
            ax6 = fig.add_subplot(gs[plot_idx, :])
            self._plot_trained_tokens(ax6)

        # Add title with metadata
        title = "MLX LoRA Training Metrics"
        if self.metrics.timestamp:
            title += f" | Run: {self.metrics.timestamp}"
        if self.metrics.trainable_params:
            title += f" | Trainable: {self.metrics.trainable_percentage:.3f}% ({self.metrics.trainable_params})"
        if self.metrics.total_iters:
            title += f" | Total Iterations: {self.metrics.total_iters}"

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

        # Save or show
        if output_path:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"✓ Saved visualization to: {output_path}")
        else:
            plt.show()

        plt.close()

    def _plot_losses(self, ax):
        """Plot training and validation losses"""
        if len(self.metrics.train_losses) > 0:
            ax.plot(
                self.metrics.iterations,
                self.metrics.train_losses,
                "o-",
                label="Training Loss",
                linewidth=2,
                markersize=4,
                alpha=0.7,
            )

        if len(self.metrics.val_losses) > 0:
            ax.plot(
                self.metrics.val_iterations,
                self.metrics.val_losses,
                "s-",
                label="Validation Loss",
                linewidth=2,
                markersize=6,
                alpha=0.7,
            )

        ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
        ax.set_title("Training Progress: Loss Curves", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add final loss annotations
        if len(self.metrics.train_losses) > 0:
            final_train = self.metrics.train_losses[-1]
            ax.text(
                0.02,
                0.98,
                f"Final Train Loss: {final_train:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=9,
            )

        if len(self.metrics.val_losses) > 0:
            final_val = self.metrics.val_losses[-1]
            ax.text(
                0.02,
                0.88,
                f"Final Val Loss: {final_val:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                fontsize=9,
            )

    def _plot_learning_rate(self, ax):
        """Plot learning rate schedule"""
        if len(self.metrics.learning_rates) > 0:
            ax.plot(
                self.metrics.iterations, self.metrics.learning_rates, "g-", linewidth=2, alpha=0.7
            )
            ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
            ax.set_ylabel("Learning Rate", fontsize=11, fontweight="bold")
            ax.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    def _plot_iterations_per_sec(self, ax):
        """Plot training speed in iterations per second"""
        if len(self.metrics.iterations_per_sec) > 0:
            ax.plot(
                self.metrics.iterations,
                self.metrics.iterations_per_sec,
                "b-",
                linewidth=2,
                alpha=0.7,
            )
            mean_speed = np.mean(self.metrics.iterations_per_sec)
            ax.axhline(
                mean_speed,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Mean: {mean_speed:.3f} it/s",
            )
            ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
            ax.set_ylabel("Iterations/sec", fontsize=11, fontweight="bold")
            ax.set_title("Training Speed (Iterations)", fontsize=12, fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")

    def _plot_tokens_per_sec(self, ax):
        """Plot throughput in tokens per second"""
        if len(self.metrics.tokens_per_sec) > 0:
            ax.plot(
                self.metrics.iterations, self.metrics.tokens_per_sec, "m-", linewidth=2, alpha=0.7
            )
            mean_throughput = np.mean(self.metrics.tokens_per_sec)
            ax.axhline(
                mean_throughput,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Mean: {mean_throughput:.1f} tok/s",
            )
            ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
            ax.set_ylabel("Tokens/sec", fontsize=11, fontweight="bold")
            ax.set_title("Training Throughput (Tokens)", fontsize=12, fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")

    def _plot_memory(self, ax):
        """Plot peak memory usage"""
        if len(self.metrics.peak_memory) > 0:
            ax.plot(self.metrics.iterations, self.metrics.peak_memory, "r-", linewidth=2, alpha=0.7)
            max_mem = max(self.metrics.peak_memory)
            ax.axhline(
                max_mem, color="darkred", linestyle="--", alpha=0.5, label=f"Max: {max_mem:.2f} GB"
            )
            ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
            ax.set_ylabel("Peak Memory (GB)", fontsize=11, fontweight="bold")
            ax.set_title("Memory Usage", fontsize=12, fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")

    def _plot_trained_tokens(self, ax):
        """Plot cumulative trained tokens"""
        if len(self.metrics.trained_tokens) > 0:
            tokens_k = [t / 1000 for t in self.metrics.trained_tokens]
            ax.plot(self.metrics.iterations, tokens_k, "c-", linewidth=2, alpha=0.7)
            final_tokens = self.metrics.trained_tokens[-1]
            ax.text(
                0.98,
                0.02,
                f"Total: {final_tokens:,} tokens",
                transform=ax.transAxes,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="cyan", alpha=0.3),
                fontsize=9,
            )
            ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
            ax.set_ylabel("Trained Tokens (K)", fontsize=11, fontweight="bold")
            ax.set_title("Token Progress", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")

    def print_summary(self):
        """Print a summary of the training metrics"""
        print("\n" + "=" * 70)
        print("MLX LoRA TRAINING SUMMARY")
        print("=" * 70)

        if self.metrics.timestamp:
            print(f"\nRun: {self.metrics.timestamp}")

        if self.metrics.log_path:
            print(f"Log: {self.metrics.log_path}")

        if self.metrics.trainable_params:
            print("\nModel Configuration:")
            print(f"  Trainable Parameters: {self.metrics.trainable_params}")
            print(f"  Trainable Percentage: {self.metrics.trainable_percentage:.3f}%")

        if self.metrics.total_iters:
            print(f"  Total Iterations: {self.metrics.total_iters}")

        if len(self.metrics.train_losses) > 0:
            print("\nTraining Loss:")
            initial_loss = self.metrics.train_losses[0]
            final_loss = self.metrics.train_losses[-1]
            print(f"  Initial: {initial_loss:.4f}")
            print(f"  Final:   {final_loss:.4f}")
            print(f"  Min:     {min(self.metrics.train_losses):.4f}")
            if initial_loss != 0:
                reduction = (1 - final_loss / initial_loss) * 100
                print(f"  Reduction: {reduction:.2f}%")
            else:
                print("  Reduction: N/A (initial loss is zero)")

        if len(self.metrics.val_losses) > 0:
            print("\nValidation Loss:")
            print(f"  Initial: {self.metrics.val_losses[0]:.4f}")
            print(f"  Final:   {self.metrics.val_losses[-1]:.4f}")
            print(f"  Min:     {min(self.metrics.val_losses):.4f}")

        if len(self.metrics.iterations_per_sec) > 0:
            print("\nTraining Speed:")
            print(f"  Mean:    {np.mean(self.metrics.iterations_per_sec):.3f} it/s")
            print(f"  Min:     {min(self.metrics.iterations_per_sec):.3f} it/s")
            print(f"  Max:     {max(self.metrics.iterations_per_sec):.3f} it/s")

        if len(self.metrics.tokens_per_sec) > 0:
            print("\nThroughput:")
            print(f"  Mean:    {np.mean(self.metrics.tokens_per_sec):.1f} tokens/s")
            print(f"  Min:     {min(self.metrics.tokens_per_sec):.1f} tokens/s")
            print(f"  Max:     {max(self.metrics.tokens_per_sec):.1f} tokens/s")

        if len(self.metrics.peak_memory) > 0:
            print("\nMemory Usage:")
            print(f"  Peak:    {max(self.metrics.peak_memory):.2f} GB")
            print(f"  Final:   {self.metrics.peak_memory[-1]:.2f} GB")

        if len(self.metrics.trained_tokens) > 0:
            print(f"\nTotal Tokens Trained: {self.metrics.trained_tokens[-1]:,}")

        print("\n" + "=" * 70 + "\n")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and visualize MLX LoRA training logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize latest training run (creates individual plot files)
  %(prog)s
  
  # Visualize specific run by timestamp
  %(prog)s --timestamp 2026-02-01_01-28-37
  
  # List all available runs
  %(prog)s --list
  
  # Generate single combined plot instead of individual files
  %(prog)s --combined
  
  # Save plots in custom subdirectory
  %(prog)s --output my_analysis
  
  # Specify custom base directory
  %(prog)s --base-dir /path/to/models/runs/mlx
  
  # High resolution output
  %(prog)s --dpi 300
        """,
    )

    parser.add_argument(
        "--timestamp",
        "-t",
        type=str,
        default=None,
        help="Specific run timestamp (format: YYYY-MM-DD_HH-MM-SS). If not provided, uses latest run.",
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        type=Path,
        default=None,
        help="Base directory for training runs (default: models/runs/mlx)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom output directory name within models/visualization/{timestamp}/plots/. Default creates numbered plot files.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate a single combined plot instead of individual files",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved figure (default: 150)")
    parser.add_argument(
        "--no-summary", action="store_true", help="Skip printing summary statistics"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List all available training runs and exit"
    )

    args = parser.parse_args()

    # Initialize run manager
    run_manager = TrainingRunManager(args.base_dir)

    # Handle list command
    if args.list:
        run_manager.list_runs()
        return 0

    # Get the training run to visualize
    run = run_manager.get_run(args.timestamp)

    if run is None:
        if args.timestamp:
            print(f"\n✗ Error: Training run not found for timestamp: {args.timestamp}")
        else:
            print(f"\n✗ Error: No training runs found in {run_manager.base_dir}")
            print("\nTip: Run training first with:")
            print("  python scripts/train_local.py")
        return 1

    timestamp, run_dir = run
    log_path = run_dir / "train.log"

    # Check training status
    print(f"\n{'=' * 70}")
    print(f"Analyzing Training Run: {timestamp}")
    print(f"{'=' * 70}")
    print(f"Run Directory: {run_dir}")
    print(f"Log File: {log_path}\n")

    success, error_msg = run_manager.check_training_status(log_path)

    if not success:
        print("✗ TRAINING FAILED")
        print(f"  Reason: {error_msg}")
        print("\nPlease check the log file for details:")
        print(f"  {log_path}")
        print()
        return 1

    print("✓ Training completed successfully\n")

    # Read log file
    print("Reading log file...")
    with open(log_path) as f:
        log_content = f.read()

    # Parse log
    print("Parsing training metrics...")
    parser_obj = MLXLogParser()
    metrics = parser_obj.parse_log(log_content, log_path)

    # Check if we got any data
    if len(metrics.iterations) == 0 and len(metrics.val_iterations) == 0:
        print("✗ Warning: No training metrics found in log file!")
        print("The log file exists but appears to be empty or invalid.")
        return 1

    print(f"✓ Found {len(metrics.iterations)} training iterations")
    print(f"✓ Found {len(metrics.val_iterations)} validation checkpoints")

    # Determine output path
    # Get project root (parent of scripts/)
    script_dir = Path(__file__).parent  # This is scripts/
    project_root = script_dir.parent  # Go up to project root

    # Use custom directory name if provided, otherwise use 'plots'
    plots_subdir = args.output if args.output else "plots"
    viz_base = project_root / "models" / "visualization" / timestamp / plots_subdir
    viz_base.mkdir(parents=True, exist_ok=True)

    # Create visualizer
    visualizer = MLXLogVisualizer(metrics)

    # Print summary
    if not args.no_summary:
        visualizer.print_summary()

    # Create plots
    print("Generating visualizations...")

    if args.combined:
        # Generate single combined plot
        output_path = viz_base / "training_metrics_combined.png"
        visualizer.create_comprehensive_plot(output_path=str(output_path), dpi=args.dpi)
        print(f"\n{'=' * 70}")
        print("✓ Visualization Complete!")
        print(f"{'=' * 70}")
        print(f"Saved to: {output_path}")
    else:
        # Generate individual plot files (default)
        saved_files = visualizer.create_individual_plots(output_dir=str(viz_base), dpi=args.dpi)
        print(f"\n{'=' * 70}")
        print("✓ Visualization Complete!")
        print(f"{'=' * 70}")
        print(f"Generated {len(saved_files)} plot files in: {viz_base}")
        print("\nFiles created:")
        for f in saved_files:
            print(f"  • {f.name}")

    print(f"\nVisualization directory: {viz_base}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
