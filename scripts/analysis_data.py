#!/usr/bin/env python3
"""
Logical Compression Analysis
Analyzes symbol usage, logical coherence, and pattern quality in compression training data.
Based on propositional logic principles: scope, precedence, ambiguity detection.
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ============================================================================
# SYMBOL DEFINITIONS
# ============================================================================

SYMBOLS = {
    "→": "implication",  # if-then, leads to, results in (\u2192)
    "|": "separator",  # separates facts/alternatives
    "@": "location",  # at, located at
    "∵": "causation",  # because, due to (\u2235)
    ":": "assignment",  # label, definition
}

# Logical operators from propositional logic
LOGICAL_CONNECTIVES = {
    "&": "conjunction",  # and
    "|": "disjunction",  # or (if used logically)
    "→": "implication",  # if-then
    "⇔": "biconditional",  # if and only if
    "¬": "negation",  # not
    "~": "negation_alt",  # not (alternative)
}


# ============================================================================
# PATTERN ANALYSIS FUNCTIONS
# ============================================================================


def extract_verbose_compressed(sample: dict) -> tuple[str, str]:
    """Extract input (verbose) and output (compressed) from message structure."""
    messages = sample.get("messages", [])

    verbose = ""
    compressed = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user" and "Compress:" in content:
            # Extract text after "Compress:"
            verbose = content.split("Compress:", 1)[1].strip()
        elif role == "assistant":
            compressed = content.strip()

    return verbose, compressed


def tokenize_compression(compressed: str) -> list[str]:
    """Break compression into tokens: symbols and text chunks."""
    tokens = []
    current = []

    for char in compressed:
        if char in SYMBOLS or char in LOGICAL_CONNECTIVES:
            if current:
                tokens.append("".join(current).strip())
                current = []
            tokens.append(char)
        else:
            current.append(char)

    if current:
        tokens.append("".join(current).strip())

    return [t for t in tokens if t]


def extract_symbol_sequence(compressed: str) -> str:
    """Extract just the symbol sequence (X for content, symbols as-is)."""
    result = []
    for char in compressed:
        if char in SYMBOLS or char in LOGICAL_CONNECTIVES:
            result.append(char)
        elif result and result[-1] != "X":
            result.append("X")

    return "".join(result)


def detect_scope_ambiguity(compressed: str) -> list[str]:
    """
    Detect scope ambiguities based on propositional logic.
    E.g., A → B | C could mean (A → B) | C or A → (B | C)
    """
    issues = []

    # Pattern: multiple implications with separator, no parentheses
    # A → B | C → D is ambiguous
    # Check if there are parentheses to disambiguate
    if (
        compressed.count("→") > 1
        and ("|" in compressed or "&" in compressed)
        and "(" not in compressed
    ):
        issues.append("SCOPE_AMBIGUITY: Multiple → with | or & but no grouping")

    # Pattern: A | B → C (which binds first?)
    if "|" in compressed and "→" in compressed:
        parts = re.split(r"[()]", compressed)
        for part in parts:
            if "|" in part and "→" in part:
                # Check order
                pipe_idx = part.find("|")
                arrow_idx = part.find("→")
                if pipe_idx < arrow_idx:
                    issues.append("PRECEDENCE_UNCLEAR: | before → without grouping")

    return issues


def detect_orphaned_symbols(compressed: str) -> list[str]:
    """Detect symbols at start/end or consecutive symbols."""
    issues = []

    # Symbol at start
    if compressed and compressed[0] in SYMBOLS:
        issues.append(f"ORPHAN_START: {compressed[0]} at beginning")

    # Symbol at end
    if compressed and compressed[-1] in SYMBOLS:
        issues.append(f"ORPHAN_END: {compressed[-1]} at end")

    # Consecutive symbols
    for i in range(len(compressed) - 1):
        if compressed[i] in SYMBOLS and compressed[i + 1] in SYMBOLS:
            issues.append(f"CONSECUTIVE: {compressed[i]}{compressed[i + 1]}")
            break

    return issues


def analyze_symbol_context(verbose: str, compressed: str, symbol: str) -> dict:
    """
    Analyze when a symbol is used vs when it could/should be used.
    """
    result = {
        "symbol": symbol,
        "used": symbol in compressed,
        "context_present": False,
        "context_keywords": [],
    }

    verbose_lower = verbose.lower()

    if symbol == "→":
        # Implication keywords
        keywords = ["if", "then", "therefore", "thus", "hence", "leads to", "results in", "causes"]
        for kw in keywords:
            if kw in verbose_lower:
                result["context_present"] = True
                result["context_keywords"].append(kw)

    elif symbol == "∵":
        # Causation keywords
        keywords = ["because", "since", "due to", "caused by", "reason", "as a result"]
        for kw in keywords:
            if kw in verbose_lower:
                result["context_present"] = True
                result["context_keywords"].append(kw)

    elif symbol == "@":
        # Location keywords
        keywords = ["at", "in", "located", "place", "where", "location", "based"]
        for kw in keywords:
            if kw in verbose_lower:
                result["context_present"] = True
                result["context_keywords"].append(kw)

    elif symbol == "|":
        # Separator - lists, alternatives
        keywords = ["and", "or", ",", ";"]
        for kw in keywords:
            if kw in verbose_lower:
                result["context_present"] = True
                result["context_keywords"].append(kw)

    elif symbol == ":":
        # Assignment/definition
        keywords = ["is", "are", "means", "defined as", "represents"]
        for kw in keywords:
            if kw in verbose_lower:
                result["context_present"] = True
                result["context_keywords"].append(kw)

    return result


def extract_logical_pattern(compressed: str) -> str:
    """
    Extract logical structure pattern.
    Similar to propositional logic form but for compression.
    """
    # Replace content with P, Q, R... but keep symbols
    pattern = compressed

    # Remove parentheses content but keep structure
    pattern = re.sub(r"\([^)]+\)", "(P)", pattern)

    # Replace text chunks with P
    pattern = re.sub(r"[^→|@∵:()]+", "P", pattern)

    # Collapse consecutive P's
    pattern = re.sub(r"P+", "P", pattern)

    return pattern.strip()


def check_negation_preservation(verbose: str, compressed: str) -> dict:
    """Check if negations are preserved."""
    negation_words = ["not", "no", "never", "neither", "nor", "n't", "without", "none"]

    verbose_lower = verbose.lower()
    compressed_lower = compressed.lower()

    verbose_has_negation = any(
        re.search(r"\b" + word + r"\b", verbose_lower) for word in negation_words
    )
    compressed_has_negation = any(
        re.search(r"\b" + word + r"\b", compressed_lower) for word in negation_words
    )

    # Check for negation symbols
    has_neg_symbol = "¬" in compressed or "~" in compressed or "!" in compressed

    return {
        "verbose_has_negation": verbose_has_negation,
        "compressed_has_negation": compressed_has_negation or has_neg_symbol,
        "negation_lost": verbose_has_negation and not (compressed_has_negation or has_neg_symbol),
    }


def analyze_symbol_combinations(compressed: str) -> list[tuple[str, str]]:
    """Extract symbol pairs (bigrams) to find common combinations."""
    symbols_only = [char for char in compressed if char in SYMBOLS]
    bigrams = []
    for i in range(len(symbols_only) - 1):
        bigrams.append((symbols_only[i], symbols_only[i + 1]))
    return bigrams


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def analyze_dataset(data: list[dict]) -> dict:
    """Run comprehensive logical analysis on compression data."""

    results = {
        "total_samples": len(data),
        "symbol_usage": defaultdict(int),
        "symbol_context_analysis": defaultdict(
            lambda: {
                "used_when_context_present": 0,
                "not_used_when_context_present": 0,
                "used_without_context": 0,
            }
        ),
        "scope_ambiguities": [],
        "orphaned_symbols": [],
        "logical_patterns": Counter(),
        "symbol_combinations": Counter(),
        "negation_analysis": {
            "total_with_negation": 0,
            "negation_preserved": 0,
            "negation_lost": 0,
        },
        "compression_ratios": [],
        "problematic_samples": [],
        "good_samples": [],
    }

    for idx, sample in enumerate(data):
        verbose, compressed = extract_verbose_compressed(sample)

        if not verbose or not compressed:
            continue

        # Compression ratio
        v_tokens = len(verbose.split())
        c_tokens = len(compressed.split())
        ratio = v_tokens / c_tokens if c_tokens > 0 else 0
        results["compression_ratios"].append(ratio)

        # Symbol usage
        for symbol in SYMBOLS:
            if symbol in compressed:
                results["symbol_usage"][symbol] += 1

        # Symbol context analysis
        for symbol in SYMBOLS:
            ctx = analyze_symbol_context(verbose, compressed, symbol)

            if ctx["context_present"] and ctx["used"]:
                results["symbol_context_analysis"][symbol]["used_when_context_present"] += 1
            elif ctx["context_present"] and not ctx["used"]:
                results["symbol_context_analysis"][symbol]["not_used_when_context_present"] += 1
            elif not ctx["context_present"] and ctx["used"]:
                results["symbol_context_analysis"][symbol]["used_without_context"] += 1

        # Scope ambiguity detection
        scope_issues = detect_scope_ambiguity(compressed)
        if scope_issues:
            results["scope_ambiguities"].append(
                {"id": idx, "issues": scope_issues, "compressed": compressed[:150]}
            )

        # Orphaned symbols
        orphan_issues = detect_orphaned_symbols(compressed)
        if orphan_issues:
            results["orphaned_symbols"].append(
                {"id": idx, "issues": orphan_issues, "compressed": compressed[:150]}
            )

        # Logical patterns
        pattern = extract_logical_pattern(compressed)
        results["logical_patterns"][pattern] += 1

        # Symbol combinations
        bigrams = analyze_symbol_combinations(compressed)
        for bg in bigrams:
            results["symbol_combinations"][bg] += 1

        # Negation analysis
        neg_check = check_negation_preservation(verbose, compressed)
        if neg_check["verbose_has_negation"]:
            results["negation_analysis"]["total_with_negation"] += 1
            if neg_check["compressed_has_negation"]:
                results["negation_analysis"]["negation_preserved"] += 1
            else:
                results["negation_analysis"]["negation_lost"] += 1

        # Flag problematic samples
        if ratio < 1.0 or scope_issues or orphan_issues:
            results["problematic_samples"].append(
                {
                    "id": idx,
                    "ratio": ratio,
                    "scope_issues": scope_issues,
                    "orphan_issues": orphan_issues,
                    "verbose": verbose[:100],
                    "compressed": compressed[:100],
                }
            )

        # Flag good samples
        if ratio > 3.0 and not scope_issues and not orphan_issues:
            results["good_samples"].append(
                {"id": idx, "ratio": ratio, "compressed": compressed[:150]}
            )

    return results


def generate_report(results: dict) -> str:
    """Generate analysis report."""

    lines = []
    lines.append("=" * 80)
    lines.append("LOGICAL COMPRESSION ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Dataset overview
    lines.append("DATASET OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"Total samples analyzed: {results['total_samples']}")
    lines.append("")

    # Compression quality
    lines.append("COMPRESSION QUALITY")
    lines.append("-" * 80)
    ratios = results["compression_ratios"]
    if ratios:
        lines.append(f"Mean ratio: {statistics.mean(ratios):.2f}x")
        lines.append(f"Median ratio: {statistics.median(ratios):.2f}x")
        lines.append(f"Min: {min(ratios):.2f}x | Max: {max(ratios):.2f}x")
        lines.append(
            f"Samples with ratio < 1.0: {sum(1 for r in ratios if r < 1.0)} (WORSE than input)"
        )
        lines.append(
            f"Samples with ratio > 3.0: {sum(1 for r in ratios if r > 3.0)} (GOOD compression)"
        )
    lines.append("")

    # Symbol usage
    lines.append("SYMBOL USAGE")
    lines.append("-" * 80)
    total = results["total_samples"]
    for symbol, name in SYMBOLS.items():
        count = results["symbol_usage"][symbol]
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"{symbol} ({name:12s}): {count:4d} / {total} ({pct:5.1f}%)")
    lines.append("")

    # Symbol context analysis (KEY INSIGHT)
    lines.append("SYMBOL CONTEXT ANALYSIS (When should symbol be used?)")
    lines.append("-" * 80)
    for symbol in SYMBOLS:
        ctx = results["symbol_context_analysis"][symbol]
        used_when_should = ctx["used_when_context_present"]
        missed_when_should = ctx["not_used_when_context_present"]
        used_wrongly = ctx["used_without_context"]

        total_opportunities = used_when_should + missed_when_should
        if total_opportunities > 0:
            accuracy = used_when_should / total_opportunities * 100
            lines.append(f"\n{symbol} ({SYMBOLS[symbol]}):")
            lines.append(
                f"  Correctly used when context present: {used_when_should} / {total_opportunities} ({accuracy:.1f}%)"
            )
            lines.append(f"  Missed opportunities: {missed_when_should}")
            lines.append(f"  Used without clear context: {used_wrongly}")
    lines.append("")

    # Symbol combinations (PATTERN DISCOVERY)
    lines.append("SYMBOL COMBINATIONS (Bigrams)")
    lines.append("-" * 80)
    lines.append("Most common symbol sequences:")
    for (s1, s2), count in results["symbol_combinations"].most_common(15):
        lines.append(f"  {s1}{s2} : {count:3d} times")
    lines.append("")

    # Logical patterns
    lines.append("TOP LOGICAL PATTERNS")
    lines.append("-" * 80)
    for pattern, count in results["logical_patterns"].most_common(20):
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"{pattern:50s} : {count:3d} ({pct:4.1f}%)")
    lines.append("")

    # Scope ambiguities (CRITICAL ISSUE)
    lines.append("SCOPE AMBIGUITIES (Propositional Logic Issues)")
    lines.append("-" * 80)
    if results["scope_ambiguities"]:
        lines.append(f"Total samples with scope ambiguity: {len(results['scope_ambiguities'])}")
        lines.append("\nFirst 5 examples:")
        for item in results["scope_ambiguities"][:5]:
            lines.append(f"\n  ID {item['id']}:")
            for issue in item["issues"]:
                lines.append(f"    - {issue}")
            lines.append(f"    Sample: {item['compressed'][:100]}...")
    else:
        lines.append("No scope ambiguities detected.")
    lines.append("")

    # Orphaned symbols
    lines.append("ORPHANED SYMBOLS (Syntax Errors)")
    lines.append("-" * 80)
    if results["orphaned_symbols"]:
        lines.append(f"Total samples with orphaned symbols: {len(results['orphaned_symbols'])}")
        lines.append("\nFirst 5 examples:")
        for item in results["orphaned_symbols"][:5]:
            lines.append(f"\n  ID {item['id']}:")
            for issue in item["issues"]:
                lines.append(f"    - {issue}")
            lines.append(f"    Sample: {item['compressed'][:100]}...")
    else:
        lines.append("No orphaned symbols detected.")
    lines.append("")

    # Negation analysis
    lines.append("NEGATION PRESERVATION")
    lines.append("-" * 80)
    neg = results["negation_analysis"]
    if neg["total_with_negation"] > 0:
        preservation_rate = neg["negation_preserved"] / neg["total_with_negation"] * 100
        lines.append(f"Samples with negation in input: {neg['total_with_negation']}")
        lines.append(f"Negation preserved: {neg['negation_preserved']} ({preservation_rate:.1f}%)")
        lines.append(f"Negation LOST: {neg['negation_lost']} ({100 - preservation_rate:.1f}%)")
    else:
        lines.append("No negations detected in inputs.")
    lines.append("")

    # Problematic samples
    lines.append("PROBLEMATIC SAMPLES (First 10)")
    lines.append("-" * 80)
    for item in results["problematic_samples"][:10]:
        lines.append(f"\nID {item['id']}: Ratio {item['ratio']:.2f}x")
        if item["scope_issues"]:
            lines.append(f"  Scope issues: {', '.join(item['scope_issues'])}")
        if item["orphan_issues"]:
            lines.append(f"  Orphan issues: {', '.join(item['orphan_issues'])}")
        lines.append(f"  Input: {item['verbose']}...")
        lines.append(f"  Output: {item['compressed']}...")
    lines.append("")

    # Good samples
    lines.append("GOOD SAMPLES (High quality compressions, first 5)")
    lines.append("-" * 80)
    for item in sorted(results["good_samples"], key=lambda x: x["ratio"], reverse=True)[:5]:
        lines.append(f"\nID {item['id']}: Ratio {item['ratio']:.2f}x")
        lines.append(f"  {item['compressed']}")
    lines.append("")

    # RECOMMENDATIONS
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 80)

    rec_num = 1

    # Symbol usage recommendations
    for symbol in ["@", "∵"]:
        ctx = results["symbol_context_analysis"][symbol]
        total_opportunities = (
            ctx["used_when_context_present"] + ctx["not_used_when_context_present"]
        )
        if total_opportunities > 0:
            accuracy = ctx["used_when_context_present"] / total_opportunities * 100
            if accuracy < 50:
                lines.append(
                    f"{rec_num}. UNDERUSED SYMBOL {symbol}: Only used {accuracy:.1f}% when context present."
                )
                lines.append(
                    f"   Action: Add training examples explicitly using {symbol} or filter samples missing it."
                )
                rec_num += 1

    # Scope ambiguity
    if len(results["scope_ambiguities"]) > total * 0.1:
        lines.append(
            f"{rec_num}. SCOPE AMBIGUITY: {len(results['scope_ambiguities'])} samples have unclear precedence."
        )
        lines.append(
            "   Action: Define operator precedence (e.g., | binds tighter than →) or require parentheses."
        )
        rec_num += 1

    # Negation loss
    if neg["total_with_negation"] > 0:
        loss_rate = neg["negation_lost"] / neg["total_with_negation"] * 100
        if loss_rate > 30:
            lines.append(f"{rec_num}. NEGATION LOST: {loss_rate:.1f}% of negations are dropped.")
            lines.append("   Action: Add negation symbol (¬ or ~) to preserve meaning.")
            rec_num += 1

    # Bad compressions
    bad_count = sum(1 for r in ratios if r < 1.0)
    if bad_count > 0:
        lines.append(
            f"{rec_num}. FILTER BAD SAMPLES: {bad_count} samples are LONGER than input (ratio < 1.0)."
        )
        lines.append("   Action: Remove these from training data.")
        rec_num += 1

    # Symbol combinations insight
    top_combo = results["symbol_combinations"].most_common(1)
    if top_combo:
        (s1, s2), count = top_combo[0]
        lines.append(f"{rec_num}. DOMINANT PATTERN: {s1}{s2} appears {count} times.")
        lines.append(
            "   Insight: This is your model's most common structure. Ensure it's logically sound."
        )
        rec_num += 1

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================


def main():
    # EDIT THESE PATHS
    INPUT_FILE = "data/training/train.jsonl"  # <-- PUT YOUR FILE PATH HERE
    OUTPUT_REPORT = "logical_analysis_report.txt"
    OUTPUT_JSON = "logical_analysis_data.json"
    OUTPUT_DIR = Path("reports/logical_analysis")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_REPORT = OUTPUT_DIR / f"logical_analysis_report_{timestamp}.txt"
    OUTPUT_JSON = OUTPUT_DIR / f"logical_analysis_data_{timestamp}.json"

    print(f"Loading data from {INPUT_FILE}...")

    try:
        with open(INPUT_FILE, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: File '{INPUT_FILE}' not found.")
        print("Please edit the INPUT_FILE path in the script.")
        return

    print(f"Loaded {len(data)} samples.")
    print("Running logical analysis...")

    results = analyze_dataset(data)

    print("Generating report...")
    report = generate_report(results)

    # Print to console
    print("\n" + report)

    # Save report
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {OUTPUT_REPORT}")

    # Save raw data
    serializable = {
        "total_samples": results["total_samples"],
        "symbol_usage": dict(results["symbol_usage"]),
        "symbol_context_analysis": {
            k: dict(v) for k, v in results["symbol_context_analysis"].items()
        },
        "compression_ratios": {
            "mean": statistics.mean(results["compression_ratios"])
            if results["compression_ratios"]
            else 0,
            "median": statistics.median(results["compression_ratios"])
            if results["compression_ratios"]
            else 0,
            "min": min(results["compression_ratios"]) if results["compression_ratios"] else 0,
            "max": max(results["compression_ratios"]) if results["compression_ratios"] else 0,
        },
        "logical_patterns": dict(results["logical_patterns"].most_common(50)),
        "symbol_combinations": {
            f"{s1}{s2}": count for (s1, s2), count in results["symbol_combinations"].most_common(30)
        },
        "negation_analysis": results["negation_analysis"],
        "scope_ambiguities_count": len(results["scope_ambiguities"]),
        "orphaned_symbols_count": len(results["orphaned_symbols"]),
        "problematic_count": len(results["problematic_samples"]),
        "good_samples_count": len(results["good_samples"]),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Raw data saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
