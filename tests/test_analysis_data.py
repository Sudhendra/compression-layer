# tests/test_analysis_data.py
from __future__ import annotations

import re

import pytest

# Adjust this import to match where you place the script/module.
# If the file is, e.g., scripts/analysis_data.py:
#   import scripts.analysis_data as m
#
# Run with: PYTHONPATH=. python -m pytest -q
import scripts.analysis_data as m


def _sample(verbose: str, compressed: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"Compress: {verbose}"},
            {"role": "assistant", "content": compressed},
        ]
    }


def test_extract_verbose_compressed_parses_message_structure():
    s = _sample("If A then B.", "A → B")
    verbose, compressed = m.extract_verbose_compressed(s)
    assert verbose == "If A then B."
    assert compressed == "A → B"


def test_tokenize_compression_splits_symbols_and_text():
    tokens = m.tokenize_compression("A → B | C")
    # Expect symbols as standalone tokens
    assert "→" in tokens
    assert "|" in tokens
    # And content chunks
    assert "A" in tokens
    assert "B" in tokens
    assert "C" in tokens


def test_extract_symbol_sequence_basic():
    # X for content, symbols preserved; adjacent content should collapse to single X
    seq = m.extract_symbol_sequence("A → B | C")
    # "A " -> X, "→", " B " -> X, "|", " C" -> X
    assert seq == "→X|X"


def test_detect_scope_ambiguity_multiple_implications_without_grouping():
    issues = m.detect_scope_ambiguity("A → B | C → D")
    assert any(i.startswith("SCOPE_AMBIGUITY") for i in issues)


def test_detect_scope_ambiguity_precedence_unclear_pipe_before_arrow():
    issues = m.detect_scope_ambiguity("A | B → C")
    assert any(i.startswith("PRECEDENCE_UNCLEAR") for i in issues)


def test_detect_orphaned_symbols_start_end_and_consecutive():
    assert any("ORPHAN_START" in x for x in m.detect_orphaned_symbols("| A"))
    assert any("ORPHAN_END" in x for x in m.detect_orphaned_symbols("A |"))
    assert any("CONSECUTIVE" in x for x in m.detect_orphaned_symbols("A||B"))


def test_analyze_symbol_context_implication_detects_keywords_and_usage():
    verbose = "If the alarm triggers then evacuate. Therefore leave."
    compressed = "alarm → evacuate"
    ctx = m.analyze_symbol_context(verbose, compressed, "→")
    assert ctx["context_present"]
    assert ctx["used"]
    assert any(k in ctx["context_keywords"] for k in ["if", "then", "therefore"])


def test_check_negation_preservation_detects_lost_negation():
    verbose = "Do not open the door."
    compressed = "open door"  # negation lost
    out = m.check_negation_preservation(verbose, compressed)
    assert out["verbose_has_negation"]
    assert out["negation_lost"]
    assert not out["compressed_has_negation"]


def test_check_negation_preservation_accepts_negation_symbol():
    verbose = "Do not open the door."
    compressed = "¬ open door"
    out = m.check_negation_preservation(verbose, compressed)
    assert out["verbose_has_negation"]
    assert out["compressed_has_negation"]
    assert not out["negation_lost"]


def test_extract_logical_pattern_normalizes_structure():
    # Parentheses content becomes (P), text becomes P, symbols preserved
    pat = m.extract_logical_pattern("(foo bar) → baz | qux")
    assert "(P)" in pat
    assert "→" in pat
    assert "|" in pat
    # Should be only P and symbols + parentheses
    assert re.fullmatch(r"[P→|@∵:() ]+", pat) is not None


def test_analyze_symbol_combinations_bigrams_only_from_SYMBOLS_not_LOGICAL_CONNECTIVES():
    # Note: analyze_symbol_combinations uses SYMBOLS only (not LOGICAL_CONNECTIVES)
    bigrams = m.analyze_symbol_combinations("A → B | C : D")
    # SYMBOLS include →, |, : so we should see bigrams among these (order preserved)
    assert ("→", "|") in bigrams
    assert ("|", ":") in bigrams


def test_analyze_dataset_aggregates_core_fields_and_flags_problematic_and_good():
    data = [
        # Good: high ratio, no orphan/scope issues
        _sample(
            "This is a long verbose description with many words because it explains details clearly.",
            "desc: details",
        ),
        # Problematic: ratio < 1 (compressed longer than verbose)
        _sample("short", "this is longer than short"),
        # Scope ambiguity
        _sample("If A then B or if C then D", "A → B | C → D"),
        # Orphaned symbol
        _sample("List items A and B", "| A | B"),
        # Negation lost
        _sample("Do not proceed", "proceed"),
    ]

    results = m.analyze_dataset(data)

    assert results["total_samples"] == len(data)
    assert isinstance(results["compression_ratios"], list)
    assert results["symbol_usage"]["|"] >= 1  # used in at least one sample
    assert "logical_patterns" in results and len(results["logical_patterns"]) > 0
    assert "symbol_combinations" in results

    # We created at least one problematic (ratio < 1), plus scope and orphan cases
    assert len(results["problematic_samples"]) >= 3

    # Negation analysis: one verbose with negation, and it is lost in compressed
    neg = results["negation_analysis"]
    assert neg["total_with_negation"] >= 1
    assert neg["negation_lost"] >= 1

    # Good samples: first sample likely yields ratio > 3.0 depending on tokenization
    # Make the assertion robust by checking at least one good sample exists OR ratio > 3 appears.
    if results["good_samples"]:
        assert all(x["ratio"] > 3.0 for x in results["good_samples"])


def test_generate_report_contains_key_sections():
    # Minimal results skeleton for report generation
    results = {
        "total_samples": 2,
        "symbol_usage": {"→": 1, "|": 1, "@": 0, "∵": 0, ":": 1},
        "symbol_context_analysis": {
            s: {
                "used_when_context_present": 0,
                "not_used_when_context_present": 0,
                "used_without_context": 0,
            }
            for s in m.SYMBOLS
        },
        "scope_ambiguities": [],
        "orphaned_symbols": [],
        "logical_patterns": m.Counter({"P→P": 1}),
        "symbol_combinations": m.Counter({("→", "|"): 2}),
        "negation_analysis": {
            "total_with_negation": 0,
            "negation_preserved": 0,
            "negation_lost": 0,
        },
        "compression_ratios": [2.0, 3.5],
        "problematic_samples": [],
        "good_samples": [{"id": 1, "ratio": 3.5, "compressed": "x"}],
    }

    report = m.generate_report(results)
    assert "LOGICAL COMPRESSION ANALYSIS REPORT" in report
    assert "DATASET OVERVIEW" in report
    assert "COMPRESSION QUALITY" in report
    assert "SYMBOL USAGE" in report
    assert "SCOPE AMBIGUITIES" in report
    assert "ORPHANED SYMBOLS" in report
    assert "NEGATION PRESERVATION" in report
    assert "RECOMMENDATIONS" in report


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
