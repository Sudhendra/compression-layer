#!/usr/bin/env python3
"""
Dataset Manager
Manages swapping between sanitized and original training datasets.

Usage:
    python dataset_manager.py --update    # Switch to sanitized dataset
    python dataset_manager.py --revert    # Switch back to original dataset
    python dataset_manager.py --status    # Check current state
    python dataset_manager.py --log       # View change history
    
    # With custom paths
    python dataset_manager.py --update \
        --train-file data/training/train.jsonl \
        --sanitized data/training/sanitized_train.jsonl
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================


def get_config(args) -> dict:
    """Build configuration from CLI arguments with sensible defaults."""

    # Use CLI args if provided, otherwise use defaults
    train_file = Path(args.train_file) if args.train_file else Path("data/training/train.jsonl")
    sanitized_file = (
        Path(args.sanitized) if args.sanitized else Path("data/training/sanitized_train.jsonl")
    )

    # Derive backup path from train file
    backup_file = train_file.parent / f"{train_file.stem}.original{train_file.suffix}"

    # State files in same directory as train file
    state_dir = train_file.parent

    return {
        # Main training file (the one used by training scripts)
        "active_train": train_file,
        # Backup of original data
        "original_backup": backup_file,
        # Sanitized data
        "sanitized_data": sanitized_file,
        # State and log files
        "state_file": state_dir / ".dataset_state.json",
        "log_file": state_dir / ".dataset_changes.log",
    }


# ============================================================================
# STATE MANAGEMENT
# ============================================================================


def load_state(config: dict) -> dict:
    """Load current dataset state."""
    if config["state_file"].exists():
        with open(config["state_file"]) as f:
            return json.load(f)

    return {
        "current": "original",  # 'original' or 'sanitized'
        "last_action": None,  # 'update' or 'revert'
        "last_change": None,
        "change_count": 0,
    }


def save_state(config: dict, state: dict):
    """Save dataset state."""
    config["state_file"].parent.mkdir(parents=True, exist_ok=True)
    with open(config["state_file"], "w") as f:
        json.dump(state, f, indent=2)


def log_change(config: dict, action: str, from_state: str, to_state: str, details: str = ""):
    """Log a dataset change."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"[{timestamp}] {action}: {from_state} → {to_state}"
    if details:
        log_entry += f" | {details}"
    log_entry += "\n"

    config["log_file"].parent.mkdir(parents=True, exist_ok=True)
    with open(config["log_file"], "a") as f:
        f.write(log_entry)

    print(f"✓ Logged: {log_entry.strip()}")


# ============================================================================
# FILE OPERATIONS
# ============================================================================


def count_samples(path):
    """
    Count number of JSON objects in a JSONL file.
    """
    path = Path(path)

    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                continue  # skip malformed lines safely

    return count


def verify_files_exist(config: dict) -> bool:
    """Verify required files exist."""
    errors = []

    if not config["active_train"].exists():
        errors.append(f"❌ Active training file not found: {config['active_train']}")

    if not config["sanitized_data"].exists():
        errors.append(f"❌ Sanitized data not found: {config['sanitized_data']}")

    if errors:
        print("\n".join(errors))
        print("\nPlease ensure sanitized data exists before updating.")
        return False

    return True


def backup_original(config: dict):
    """Create backup of original data if it doesn't exist."""
    if not config["original_backup"].exists() and config["active_train"].exists():
        print("Creating backup of original data...")
        shutil.copy2(config["active_train"], config["original_backup"])
        print(f"✓ Backup created: {config['original_backup']}")
        return True
    return False


# ============================================================================
# MAIN OPERATIONS
# ============================================================================


def update_to_sanitized(config: dict) -> bool:
    """Switch to sanitized dataset."""
    state = load_state(config)

    # Safety check: if last action was update, only allow revert
    if state.get("last_action") == "update":
        print("❌ Last action was already UPDATE.")
        print("   You can only REVERT after an update.")
        print()
        print("To revert to original dataset, run:")
        print(f"  python dataset_manager.py --revert --train-file {config['active_train']}")
        return False

    if state["current"] == "sanitized":
        print("⚠ Already using sanitized dataset. No changes made.")
        return False

    if not verify_files_exist(config):
        return False

    # Create backup of original if needed
    backup_original(config)

    # Get sample counts
    original_count = count_samples(config["active_train"])
    sanitized_count = count_samples(config["sanitized_data"])

    print("\nSwitching to sanitized dataset...")
    print(f"  Original samples:  {original_count}")
    print(f"  Sanitized samples: {sanitized_count}")
    print(f"  Removed samples:   {original_count - sanitized_count}")

    # Perform swap
    try:
        shutil.copy2(config["sanitized_data"], config["active_train"])

        # Update state
        state["current"] = "sanitized"
        state["last_action"] = "update"
        state["last_change"] = datetime.now().isoformat()
        state["change_count"] += 1
        save_state(config, state)

        # Log change
        log_change(
            config,
            action="UPDATE",
            from_state="original",
            to_state="sanitized",
            details=f"{original_count} → {sanitized_count} samples",
        )

        print("\n✓ Successfully switched to sanitized dataset")
        print(f"✓ {config['active_train']} now contains {sanitized_count} samples")
        print()
        print("⚠ To undo this change, run:")
        print(f"  python dataset_manager.py --revert --train-file {config['active_train']}")
        return True

    except Exception as e:
        print(f"\n❌ Error during update: {e}")
        return False


def revert_to_original(config: dict) -> bool:
    """Revert to original dataset."""
    state = load_state(config)

    # Safety check: if last action was revert, only allow update
    if state.get("last_action") == "revert":
        print("❌ Last action was already REVERT.")
        print("   You can only UPDATE after a revert.")
        print()
        print("To switch to sanitized dataset, run:")
        print(f"  python dataset_manager.py --update --train-file {config['active_train']}")
        return False

    if state["current"] == "original":
        print("⚠ Already using original dataset. No changes made.")
        return False

    if not config["original_backup"].exists():
        print(f"❌ Original backup not found: {config['original_backup']}")
        print("Cannot revert without backup.")
        return False

    # Get sample counts
    sanitized_count = count_samples(config["active_train"])
    original_count = count_samples(config["original_backup"])

    print("\nReverting to original dataset...")
    print(f"  Sanitized samples: {sanitized_count}")
    print(f"  Original samples:  {original_count}")

    # Perform swap
    try:
        shutil.copy2(config["original_backup"], config["active_train"])

        # Update state
        state["current"] = "original"
        state["last_action"] = "revert"
        state["last_change"] = datetime.now().isoformat()
        state["change_count"] += 1
        save_state(config, state)

        # Log change
        log_change(
            config,
            action="REVERT",
            from_state="sanitized",
            to_state="original",
            details=f"{sanitized_count} → {original_count} samples",
        )

        print("\n✓ Successfully reverted to original dataset")
        print(f"✓ {config['active_train']} now contains {original_count} samples")
        print()
        print("⚠ To switch back to sanitized, run:")
        print(f"  python dataset_manager.py --update --train-file {config['active_train']}")
        return True

    except Exception as e:
        print(f"\n❌ Error during revert: {e}")
        return False


def show_status(config: dict):
    """Show current dataset status."""
    state = load_state(config)

    print("\n" + "=" * 80)
    print("DATASET STATUS")
    print("=" * 80)
    print()

    # Configuration
    print("Configuration:")
    print(f"  Train file:         {config['active_train']}")
    print(f"  Sanitized file:     {config['sanitized_data']}")
    print(f"  Backup file:        {config['original_backup']}")
    print()

    # Current state
    current = state["current"].upper()
    last_action = state.get("last_action", "None")
    print(f"Current dataset:    {current}")
    print(f"Last action:        {last_action.upper() if last_action else 'None'}")
    print(f"Total changes:      {state['change_count']}")

    if state["last_change"]:
        last_change = datetime.fromisoformat(state["last_change"])
        print(f"Last change:        {last_change.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Last change:        Never")

    print()

    # File information
    print("Files:")

    if config["active_train"].exists():
        active_count = count_samples(config["active_train"])
        print(f"  ✓ {config['active_train'].name:25s} {active_count:4d} samples (ACTIVE)")
    else:
        print(f"  ❌ {config['active_train'].name:25s} Not found")

    if config["original_backup"].exists():
        original_count = count_samples(config["original_backup"])
        print(f"  ✓ {config['original_backup'].name:25s} {original_count:4d} samples (backup)")
    else:
        print(
            f"  ⚠ {config['original_backup'].name:25s} Not found (will be created on first update)"
        )

    if config["sanitized_data"].exists():
        sanitized_count = count_samples(config["sanitized_data"])
        print(f"  ✓ {config['sanitized_data'].name:25s} {sanitized_count:4d} samples")
    else:
        print(f"  ❌ {config['sanitized_data'].name:25s} Not found")

    print()

    # Recommendations
    print("=" * 80)
    print("AVAILABLE ACTIONS")
    print("=" * 80)
    print()

    # Show only the allowed action based on last action
    if last_action == "update":
        print("✓ You can REVERT (last action was UPDATE):")
        print(f"  python dataset_manager.py --revert --train-file {config['active_train']}")
        print()
        print("✗ You cannot UPDATE again (already updated)")
    elif last_action == "revert":
        print("✓ You can UPDATE (last action was REVERT):")
        print(f"  python dataset_manager.py --update --train-file {config['active_train']}")
        print()
        print("✗ You cannot REVERT again (already reverted)")
    else:
        # No previous action - allow either
        if state["current"] == "original":
            print("✓ You can UPDATE to sanitized dataset:")
            print(f"  python dataset_manager.py --update --train-file {config['active_train']}")
        else:
            print("✓ You can REVERT to original dataset:")
            print(f"  python dataset_manager.py --revert --train-file {config['active_train']}")

    print()
    print("To view change history:")
    print("  python dataset_manager.py --log")
    print()


def show_log(config: dict, lines: int | None = None):
    """Show change log."""
    if not config["log_file"].exists():
        print("No changes logged yet.")
        return

    print("\n" + "=" * 80)
    print("DATASET CHANGE LOG")
    print("=" * 80)
    print()

    with open(config["log_file"]) as f:
        log_lines = f.readlines()

    # Show last N lines if specified
    if lines:
        log_lines = log_lines[-lines:]

    if not log_lines:
        print("No changes logged yet.")
        return

    for line in log_lines:
        print(line.rstrip())

    print()
    print(f"Total entries: {len(log_lines)}")
    print()


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Manage dataset switching between sanitized and original training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Switch to sanitized dataset (default paths)
  python dataset_manager.py --update
  
  # Switch with custom paths
  python dataset_manager.py --update \\
      --train-file data/training/train.jsonl \\
      --sanitized data/training/sanitized_train.jsonl
  
  # Revert to original dataset
  python dataset_manager.py --revert --train-file data/training/train.jsonl
  
  # Check current status
  python dataset_manager.py --status
  
  # View full change log
  python dataset_manager.py --log
  
  # View last 10 changes
  python dataset_manager.py --log --lines 10

Safety:
  - After UPDATE, you can only REVERT
  - After REVERT, you can only UPDATE
  - This prevents accidental double-operations
        """,
    )

    # Actions
    parser.add_argument("--update", action="store_true", help="Switch to sanitized dataset")
    parser.add_argument("--revert", action="store_true", help="Revert to original dataset")
    parser.add_argument("--status", action="store_true", help="Show current dataset status")
    parser.add_argument("--log", action="store_true", help="Show change log")
    parser.add_argument(
        "--lines", type=int, metavar="N", help="Show last N log entries (use with --log)"
    )

    # Path configuration
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Path to training file to manage (default: data/training/train.jsonl)",
    )
    parser.add_argument(
        "--sanitized",
        type=str,
        default=None,
        help="Path to sanitized data file (default: data/training/sanitized_train.jsonl)",
    )

    args = parser.parse_args()

    # Build configuration from args
    config = get_config(args)

    # Execute requested action
    if args.update:
        update_to_sanitized(config)
    elif args.revert:
        revert_to_original(config)
    elif args.log:
        show_log(config, args.lines)
    elif args.status:
        show_status(config)
    else:
        # Default: show status
        show_status(config)


if __name__ == "__main__":
    main()
