#!/usr/bin/env python3
"""
Data Sanitization for Compression Training 

SCOPE: This script is designed specifically for data/training/train.jsonl
       with chat-message format (system/user/assistant roles).

Usage:
    # Default paths
    python data_sanitization.py
    
    # Custom paths
    python data_sanitization.py \
        --input data/training/train.jsonl \
        --sanitized data/training/sanitized_train.jsonl \
        --unsanitized data/training/unsanitized_train.jsonl

CHANGES FROM V3:
- Rule B now code-aware: allows leading @ for code samples (decorators)
- Token-based compression ratio aligned with src/utils/tokenizers.py
- Explicit format validation with parse error logging
- Guards for unexpected message structures
- CLI arguments for flexible path configuration

Extracts BOTH sanitized and unsanitized samples in one pass.
Unsanitized samples are saved for recovery analysis.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Import tokenizer for token-based ratios
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.tokenizers import compression_ratio

# ============================================================================
# SYMBOL DEFINITIONS
# ============================================================================

SYMBOLS = {"→", "|", "@", "∵", ":"}

# Strict keywords for natural language only
LOCATION_KEYWORDS_NL = [
    "located in",
    "located at",
    "based in",
    "situated in",
    "found in",
    "positioned at",
    "positioned in",
    "place is",
    "place was",
    "city of",
    "town of",
    "on the shores of",
    "near the",
    "by the",
]

CAUSATION_KEYWORDS_NL = [
    "because of",
    "due to",
    "caused by",
    "as a result of",
    "leads to",
    "results in",
    "led to",
    "resulted in",
    "owing to",
    "on account of",
    "thanks to",
    "consequently",
    "therefore",
    "thus",
]

NEGATION_KEYWORDS = [
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "n't",
    "without",
    "none",
    "nothing",
    "nobody",
    "nowhere",
    "no longer",
    "no more",
]

# Code detection indicators
CODE_INDICATORS = [
    "def ",
    "class ",
    "import ",
    "return ",
    "yield ",
    "async ",
    "await ",
    "self.",
    "__init__",
    "__",
    "lambda ",
    "isinstance(",
    "raise ",
    "@classmethod",
    "@staticmethod",
    "@property",
    "function ",
    "const ",
    "let ",
    "var ",
    "=>",
    "async function",
    "fn:",
    "->",
    "fn(",
    "void ",
    "int ",
    "string ",
    "bool ",
    "```",
    "```python",
    "```javascript",
    "```java",
    "    def ",
    "    class ",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def is_code_sample(verbose: str) -> bool:
    """
    Detect if sample is code-related.

    Uses multiple signals to avoid misclassifying brace-heavy text (JSON, etc.) as code.
    Requires at least 2 strong signals OR 1 very strong signal.
    """
    verbose_lower = verbose.lower()

    # Very strong signals (definitive code indicators)
    very_strong_indicators = [
        "def ",
        "class ",
        "function ",
        "import ",
        "return ",
        "async ",
        "await ",
        "yield ",
        "@property",
        "@staticmethod",
        "@classmethod",
        "fn:",
        "lambda ",
        "isinstance(",
        "raise ",
    ]

    for indicator in very_strong_indicators:
        if indicator.lower() in verbose_lower:
            return True  # Single very strong signal is enough

    # Strong signals (likely code)
    strong_signals = 0

    # Check for code-specific keywords
    code_keywords = [
        "self.",
        "__init__",
        "const ",
        "let ",
        "var ",
        "void ",
        "int ",
        "string ",
        "bool ",
    ]
    for keyword in code_keywords:
        if keyword.lower() in verbose_lower:
            strong_signals += 1
            break  # Count once

    # Check for indentation pattern (multiple indented lines)
    lines = verbose.split("\n")
    indented_lines = sum(1 for line in lines if line.startswith("    ") or line.startswith("\t"))
    if indented_lines >= 3:  # Increased threshold from 2 to 3
        strong_signals += 1

    # Check for code block markers
    if (
        "```python" in verbose_lower
        or "```javascript" in verbose_lower
        or "```java" in verbose_lower
    ):
        strong_signals += 1

    # Tightened code patterns - require more context
    code_patterns = [
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+\s*[\(:]",
        r"\bfunction\s+\w+\s*\(",
        r"\w+\s*=\s*function\s*\(",
        # Tightened type annotation pattern - require multiple
        r"(\w+\s*:\s*\w+.*){2,}",  # At least 2 type annotations
    ]

    for pattern in code_patterns:
        if re.search(pattern, verbose):
            strong_signals += 1
            break  # Count once

    # Tightened {...} check - only count if it looks like actual code block
    # Must have semicolons, return statements, or assignment inside braces
    brace_code_pattern = r"\{[^}]*(;|return\s|=\s)[^}]*\}"
    if re.search(brace_code_pattern, verbose):
        strong_signals += 1

    # Require at least 2 strong signals to classify as code
    return strong_signals >= 2


def extract_verbose_compressed(
    sample: dict, sample_id: int
) -> tuple[str | None, str | None, str | None]:
    """
    Extract input (verbose) and output (compressed) from chat message structure.

    EXPECTED FORMAT for data/training/train.jsonl:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Compress:\n<text>"},
            {"role": "assistant", "content": "<compressed>"}
        ]
    }

    Returns:
        (verbose, compressed, error_message)
        error_message is None if parsing succeeded
    """
    # Validate top-level structure
    if not isinstance(sample, dict):
        return None, None, f"Sample {sample_id}: Not a dict"

    if "messages" not in sample:
        return None, None, f"Sample {sample_id}: Missing 'messages' key"

    messages = sample.get("messages", [])

    if not isinstance(messages, list):
        return None, None, f"Sample {sample_id}: 'messages' is not a list"

    if len(messages) < 2:
        return (
            None,
            None,
            f"Sample {sample_id}: Expected at least 2 messages (user + assistant), got {len(messages)}",
        )

    verbose = ""
    compressed = ""
    found_user = False
    found_assistant = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            found_user = True
            if "Compress:" in content:
                verbose = content.split("Compress:", 1)[1].strip()
            else:
                return (
                    None,
                    None,
                    f"Sample {sample_id}: User message doesn't contain 'Compress:' marker",
                )

        elif role == "assistant":
            found_assistant = True
            compressed = content.strip()

    # Validate we found expected roles
    if not found_user:
        return None, None, f"Sample {sample_id}: No user message found"

    if not found_assistant:
        return None, None, f"Sample {sample_id}: No assistant message found"

    if not verbose:
        return None, None, f"Sample {sample_id}: Empty verbose text after 'Compress:'"

    if not compressed:
        return None, None, f"Sample {sample_id}: Empty compressed text"

    return verbose, compressed, None


def compute_compression_ratio_tokens(verbose: str, compressed: str) -> float:
    """
    Compute token-based compression ratio aligned with src/utils/tokenizers.py.

    Returns compression_ratio where ratio > 1.0 means expansion (bad).
    For Rule A, we want ratio <= 1.0 (compressed <= verbose in tokens).
    """
    return compression_ratio(verbose, compressed)


# ============================================================================
# VALIDATION RULES
# ============================================================================


def rule_a_ratio_check(verbose: str, compressed: str) -> tuple[bool, str]:
    """
    Rule A: Remove samples with compression ratio > 1.0 (expansion).

    Uses token-based ratio from src/utils/tokenizers.py.
    Ratio > 1.0 means compressed text is longer than input (bad compression).
    """
    ratio = compute_compression_ratio_tokens(verbose, compressed)

    # compression_ratio returns compressed/verbose
    # We want compressed <= verbose, so ratio <= 1.0
    if ratio > 1.0:
        return False, f"Ratio {ratio:.2f} > 1.0 (expansion)"
    return True, ""


def rule_b_orphaned_symbols(compressed: str, is_code: bool) -> tuple[bool, str]:
    """
    Rule B: Remove samples with orphaned symbols.

    CODE-AWARE: Allows leading @ for code samples (Python decorators).
    """
    if not compressed:
        return False, "Empty compression"

    # Check leading symbol - allow @ for code samples (decorators)
    if compressed[0] in SYMBOLS:
        if is_code and compressed[0] == "@":
            # Valid decorator pattern
            pass
        else:
            return False, f"Orphaned symbol at start: '{compressed[0]}'"

    # Check trailing symbol (: is allowed at end)
    if compressed[-1] in SYMBOLS and compressed[-1] != ":":
        return False, f"Orphaned symbol at end: '{compressed[-1]}'"

    # Check consecutive symbols (except ::)
    for i in range(len(compressed) - 1):
        if compressed[i] in SYMBOLS and compressed[i + 1] in SYMBOLS:
            if compressed[i] == ":" and compressed[i + 1] == ":":
                continue  # :: is allowed (namespace separator)
            return False, f"Consecutive symbols: '{compressed[i]}{compressed[i + 1]}'"

    return True, ""


def rule_c_negation_preservation(verbose: str, compressed: str) -> tuple[bool, str]:
    """Rule C: Remove samples that lost negation (NL only)"""
    verbose_lower = verbose.lower()
    compressed_lower = compressed.lower()

    has_input_negation = any(
        re.search(r"\b" + re.escape(kw) + r"\b", verbose_lower) for kw in NEGATION_KEYWORDS
    )

    if not has_input_negation:
        return True, ""

    has_output_negation = any(
        re.search(r"\b" + re.escape(kw) + r"\b", compressed_lower) for kw in NEGATION_KEYWORDS
    )

    has_neg_symbol = "¬" in compressed or "~" in compressed or "!" in compressed

    if not (has_output_negation or has_neg_symbol):
        return False, "Negation lost"

    return True, ""


def rule_d_semantic_symbol_usage_nl(verbose: str, compressed: str) -> tuple[bool, str]:
    """Rule D: Remove samples that should use @ or ∵ but don't (NL only)"""
    verbose_lower = verbose.lower()

    has_location_context = any(kw in verbose_lower for kw in LOCATION_KEYWORDS_NL)
    if has_location_context and "@" not in compressed:
        return False, "Location context but no '@'"

    has_causation_context = any(kw in verbose_lower for kw in CAUSATION_KEYWORDS_NL)
    if has_causation_context and "∵" not in compressed:
        return False, "Causation context but no '∵'"

    return True, ""


# ============================================================================
# MAIN SANITIZATION + EXTRACTION
# ============================================================================


def sanitize_and_extract(input_path: Path, sanitized_path: Path, unsanitized_path: Path) -> dict:
    """
    Single-pass processing: sanitize AND extract unsanitized samples.
    Both outputs maintain original JSON structure.

    SCOPE: Designed for data/training/train.jsonl with chat-message format.
    """
    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Validate expected location
    if input_path.name != "train.jsonl":
        print(f"⚠ WARNING: Expected train.jsonl, got {input_path.name}")
        print("  This script is designed for data/training/train.jsonl format")

    print(f"Loading data from {input_path}...")

    with open(input_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"✓ Loaded {len(data)} samples\n")

    stats = {
        "total_input": len(data),
        "code_samples": 0,
        "nl_samples": 0,
        "code_passed": 0,
        "nl_passed": 0,
        "rule_a_failed": 0,
        "rule_b_failed": 0,
        "rule_c_failed": 0,
        "rule_d_failed": 0,
        "parse_errors": 0,
        "passed_all": 0,
        "failed_all": 0,
        "failed_samples": [],
        "passed_samples": [],
        "parse_error_samples": [],
    }

    sanitized_data = []
    unsanitized_data = []

    print("Processing samples...\n")

    for idx, sample in enumerate(data):
        # Extract with format validation
        verbose, compressed, parse_error = extract_verbose_compressed(sample, idx)

        # Handle parse errors - tracked separately from rule failures
        # Parse errors are NOT counted as Rule A failures
        if parse_error:
            stats["parse_errors"] += 1
            stats["failed_all"] += 1
            unsanitized_data.append(sample)
            stats["parse_error_samples"].append({"id": idx, "error": parse_error, "sample": sample})
            print(f"⚠ {parse_error}")
            continue  # Skip rule validation for malformed samples

        is_code = is_code_sample(verbose)
        content_type = "code" if is_code else "nl"

        if is_code:
            stats["code_samples"] += 1
        else:
            stats["nl_samples"] += 1

        # Apply all rules
        passed = True
        failure_reason = ""
        failed_rules = []

        # Rule A (universal) - token-based ratio
        rule_a_pass, reason = rule_a_ratio_check(verbose, compressed)
        if not rule_a_pass:
            stats["rule_a_failed"] += 1
            passed = False
            failure_reason = f"Rule A: {reason}"
            failed_rules.append("A")

        # Rule B (universal, code-aware)
        if passed:
            rule_b_pass, reason = rule_b_orphaned_symbols(compressed, is_code)
            if not rule_b_pass:
                stats["rule_b_failed"] += 1
                passed = False
                failure_reason = f"Rule B: {reason}"
                failed_rules.append("B")

        # Rule C (NL only)
        if passed and not is_code:
            rule_c_pass, reason = rule_c_negation_preservation(verbose, compressed)
            if not rule_c_pass:
                stats["rule_c_failed"] += 1
                passed = False
                failure_reason = f"Rule C: {reason}"
                failed_rules.append("C")

        # Rule D (NL only)
        if passed and not is_code:
            rule_d_pass, reason = rule_d_semantic_symbol_usage_nl(verbose, compressed)
            if not rule_d_pass:
                stats["rule_d_failed"] += 1
                passed = False
                failure_reason = f"Rule D: {reason}"
                failed_rules.append("D")

        # Sort into sanitized or unsanitized
        if passed:
            stats["passed_all"] += 1
            if is_code:
                stats["code_passed"] += 1
            else:
                stats["nl_passed"] += 1

            sanitized_data.append(sample)
            stats["passed_samples"].append(
                {
                    "id": idx,
                    "type": content_type,
                    "ratio": compute_compression_ratio_tokens(verbose, compressed),
                }
            )
        else:
            stats["failed_all"] += 1
            unsanitized_data.append(sample)
            stats["failed_samples"].append(
                {
                    "id": idx,
                    "type": content_type,
                    "reason": failure_reason,
                    "failed_rules": failed_rules,
                    "sample": sample,
                }
            )

    # Save sanitized
    print(f"\nSaving sanitized data to {sanitized_path}...")
    sanitized_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sanitized_path, "w", encoding="utf-8") as f:
        for sample in sanitized_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(sanitized_data)} sanitized samples")

    # Save unsanitized
    print(f"Saving unsanitized data to {unsanitized_path}...")
    unsanitized_path.parent.mkdir(parents=True, exist_ok=True)
    with open(unsanitized_path, "w", encoding="utf-8") as f:
        for sample in unsanitized_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(unsanitized_data)} unsanitized samples\n")

    return stats


def print_statistics(stats: dict):
    """Print statistics."""
    print("=" * 80)
    print("PROCESSING STATISTICS")
    print("=" * 80)
    print()

    print(f"Total input:                {stats['total_input']:5d}")
    print(
        f"  Code samples:             {stats['code_samples']:5d} ({stats['code_samples'] / stats['total_input'] * 100:5.1f}%)"
    )
    print(
        f"  NL samples:               {stats['nl_samples']:5d} ({stats['nl_samples'] / stats['total_input'] * 100:5.1f}%)"
    )
    print()

    print(
        f"✓ SANITIZED (passed):       {stats['passed_all']:5d} ({stats['passed_all'] / stats['total_input'] * 100:5.1f}%)"
    )
    print(
        f"  Code:                     {stats['code_passed']:5d} ({stats['code_passed'] / stats['code_samples'] * 100 if stats['code_samples'] > 0 else 0:5.1f}%)"
    )
    print(
        f"  NL:                       {stats['nl_passed']:5d} ({stats['nl_passed'] / stats['nl_samples'] * 100 if stats['nl_samples'] > 0 else 0:5.1f}%)"
    )
    print()

    print(
        f"✗ UNSANITIZED (failed):     {stats['failed_all']:5d} ({stats['failed_all'] / stats['total_input'] * 100:5.1f}%)"
    )
    print()

    print("Failed by rule:")
    print(f"  Rule A (ratio > 1.0):     {stats['rule_a_failed']:5d}")
    print(f"  Rule B (orphaned symbols):{stats['rule_b_failed']:5d}")
    print(f"  Rule C (lost negation):   {stats['rule_c_failed']:5d}")
    print(f"  Rule D (missing @ or ∵):  {stats['rule_d_failed']:5d}")
    print(f"  Parse errors:             {stats['parse_errors']:5d}")
    print()

    # Show parse errors if any
    if stats["parse_errors"] > 0:
        print("=" * 80)
        print("PARSE ERRORS (First 5)")
        print("=" * 80)
        print()
        for item in stats["parse_error_samples"][:5]:
            print(f"Sample {item['id']}:")
            print(f"  Error: {item['error']}")
            print()

    # Show sample failures
    print("=" * 80)
    print("UNSANITIZED SAMPLES (First 5)")
    print("=" * 80)
    print()

    for item in stats["failed_samples"][:5]:
        print(f"Sample {item['id']} ({item.get('type', 'unknown').upper()}):")
        print(f"  Reason: {item['reason']}")
        print(f"  Failed rules: {', '.join(item.get('failed_rules', []))}")
        print()


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize compression training data with validation rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python data_sanitization.py
  
  # Custom paths
  python data_sanitization.py \\
      --input data/training/train.jsonl \\
      --sanitized data/training/sanitized_train.jsonl \\
      --unsanitized data/training/unsanitized_train.jsonl
  
  # Different dataset location
  python data_sanitization.py \\
      --input data/experiments/custom_train.jsonl \\
      --sanitized data/experiments/custom_sanitized.jsonl \\
      --unsanitized data/experiments/custom_unsanitized.jsonl

Validation Rules:
  Rule A: Compression ratio > 1.0 (expansion)
  Rule B: Orphaned symbols (code-aware for @ decorators)
  Rule C: Lost negation (NL only)
  Rule D: Missing semantic symbols @ or ∵ (NL only)
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/training/train.jsonl",
        help="Path to input training file (default: data/training/train.jsonl)",
    )
    parser.add_argument(
        "--sanitized",
        type=str,
        default="data/training/sanitized_train.jsonl",
        help="Path to output sanitized file (default: data/training/sanitized_train.jsonl)",
    )
    parser.add_argument(
        "--unsanitized",
        type=str,
        default="data/training/unsanitized_train.jsonl",
        help="Path to output unsanitized file (default: data/training/unsanitized_train.jsonl)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_path = Path(args.input)
    sanitized_path = Path(args.sanitized)
    unsanitized_path = Path(args.unsanitized)

    print("\n" + "=" * 80)
    print("DATA SANITIZATION + EXTRACTION (Single Pass)")
    print("SCOPE: data/training/train.jsonl with chat-message format")
    print("=" * 80)
    print()
    print(f"Input:       {input_path}")
    print(f"Sanitized:   {sanitized_path}")
    print(f"Unsanitized: {unsanitized_path}")
    print("=" * 80)
    print()

    # Process
    stats = sanitize_and_extract(input_path, sanitized_path, unsanitized_path)

    # Print stats
    print_statistics(stats)

    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print()
    print(f"1. {sanitized_path}")
    print(f"   → {stats['passed_all']} samples (clean, ready for training)")
    print()
    print(f"2. {unsanitized_path}")
    print(f"   → {stats['failed_all']} samples (for recovery analysis)")
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Train on sanitized data:")
    print(f"   Use {sanitized_path} ({stats['passed_all']} samples)")
    print()
    print("2. Analyze unsanitized samples for recovery:")
    print(f"   Use {unsanitized_path} ({stats['failed_all']} samples)")
    print()


if __name__ == "__main__":
    main()
