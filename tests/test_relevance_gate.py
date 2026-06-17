"""Unit tests for the deterministic recall relevance gate (utils/relevance.py).

Pure logic — no Redis, no network, no LLM. Run:
  uv run pytest tests/test_relevance_gate.py -v
"""

from agent_memory_server.utils.relevance import (
    GateOutcome,
    apply_relevance_gate,
    evaluate_result,
    extract_salient_terms,
    record_references_any_term,
)


# --- extract_salient_terms -------------------------------------------------


def test_extract_lowercases_and_filters_stopwords():
    terms = extract_salient_terms("The Mortgage rate for Chris was high")
    # "the", "for", "was" are stopwords; "high" kept; short words gone.
    assert "mortgage" in terms
    assert "rate" in terms
    assert "chris" in terms
    assert "high" in terms
    assert "the" not in terms
    assert "for" not in terms
    assert "was" not in terms


def test_extract_drops_short_tokens():
    terms = extract_salient_terms("a to of in mortgage")
    assert terms == {"mortgage"}


def test_extract_handles_none_and_empty():
    assert extract_salient_terms(None) == set()
    assert extract_salient_terms("") == set()
    assert extract_salient_terms("   ") == set()


def test_extract_unicode_accents_preserved():
    terms = extract_salient_terms("José closed the Almería deal")
    assert "josé" in terms
    assert "almería" in terms


def test_extract_is_whole_word_no_substring_leak():
    # "rich" and "enriched" tokenize separately — no substring relationship.
    terms = extract_salient_terms("enriched data")
    assert "rich" not in terms
    assert "enriched" in terms


# --- record_references_any_term --------------------------------------------


def test_reference_true_on_overlap():
    qterms = extract_salient_terms("mortgage rate question")
    assert record_references_any_term("the mortgage was approved", qterms) is True


def test_reference_false_on_no_overlap():
    qterms = extract_salient_terms("mortgage rate")
    assert record_references_any_term("the dog ate dinner", qterms) is False


def test_reference_no_substring_match():
    # query term "ben" must NOT match record "Bennett" (FIN-582 substring class).
    qterms = extract_salient_terms("ben said hello")
    assert "ben" in qterms
    assert record_references_any_term("Kevin Bennett approved it", qterms) is False


def test_reference_fail_open_when_no_query_terms():
    # All-stopword query → no anchor → fail open (keep), never silently empty.
    assert record_references_any_term("anything at all", set()) is True


# --- evaluate_result -------------------------------------------------------


def test_evaluate_strong_distance_always_kept():
    qterms = extract_salient_terms("mortgage")
    # No term overlap, but distance below floor → kept as strong semantic match.
    d = evaluate_result("the dog ate dinner", dist=0.10, query_terms=qterms, distance_floor=0.25)
    assert d.kept is True
    assert d.dropped is False
    assert d.reason == "strong-distance"


def test_evaluate_term_overlap_kept_above_floor():
    qterms = extract_salient_terms("mortgage rate")
    d = evaluate_result("mortgage approved", dist=0.40, query_terms=qterms, distance_floor=0.25)
    assert d.kept is True
    assert d.reason == "term-overlap"


def test_evaluate_no_anchor_dropped():
    qterms = extract_salient_terms("mortgage rate")
    d = evaluate_result("the dog ate dinner", dist=0.40, query_terms=qterms, distance_floor=0.25)
    assert d.kept is False
    assert d.dropped is True
    assert d.reason == "no-anchor"


def test_evaluate_floor_boundary_is_inclusive():
    qterms = extract_salient_terms("mortgage")
    # dist == floor → strong (<=), kept.
    d = evaluate_result("unrelated text", dist=0.25, query_terms=qterms, distance_floor=0.25)
    assert d.reason == "strong-distance"


def test_evaluate_no_query_terms_keeps_all():
    d = evaluate_result("anything", dist=0.9, query_terms=set(), distance_floor=0.25)
    assert d.kept is True
    assert d.reason == "no-query-terms"


# --- apply_relevance_gate --------------------------------------------------

QUERY = "what is the mortgage rate"
ITEMS = [
    ("mortgage rate is 6.5 percent", 0.12),   # strong + overlap → keep
    ("the dog ate dinner", 0.45),             # weak + no anchor → drop
    ("rate discussion with the bank", 0.40),  # weak + overlap("rate") → keep
    ("unrelated trivia", 0.10),               # strong distance → keep
]


def test_gate_disabled_is_noop():
    out = apply_relevance_gate(ITEMS, QUERY, enabled=False, shadow=False, distance_floor=0.25)
    assert out.kept_indices == (0, 1, 2, 3)
    assert out.dropped_indices == ()


def test_gate_enabled_drops_weak_no_anchor():
    out = apply_relevance_gate(ITEMS, QUERY, enabled=True, shadow=False, distance_floor=0.25)
    assert out.dropped_indices == (1,)          # only "the dog ate dinner"
    assert out.kept_indices == (0, 2, 3)
    assert out.dropped_count == 1


def test_gate_shadow_keeps_all_but_reports_drops():
    out = apply_relevance_gate(ITEMS, QUERY, enabled=True, shadow=True, distance_floor=0.25)
    # shadow keeps everything in kept_indices...
    assert out.kept_indices == (0, 1, 2, 3)
    # ...but still reports what WOULD drop.
    assert out.dropped_indices == (1,)
    assert out.dropped_count == 1


def test_gate_preserves_search_order_in_kept():
    out = apply_relevance_gate(ITEMS, QUERY, enabled=True, shadow=False, distance_floor=0.25)
    assert list(out.kept_indices) == sorted(out.kept_indices)


def test_gate_empty_items():
    out = apply_relevance_gate([], QUERY, enabled=True, shadow=False, distance_floor=0.25)
    assert out == GateOutcome(kept_indices=(), dropped_indices=(), shadow=False)


def test_gate_all_stopword_query_keeps_all():
    # Query with no salient terms → gate cannot anchor → keep everything.
    out = apply_relevance_gate(ITEMS, "the and for was", enabled=True, shadow=False, distance_floor=0.25)
    assert out.kept_indices == (0, 1, 2, 3)
    assert out.dropped_indices == ()


def test_gate_aggressive_floor_still_keeps_overlap():
    # floor=0.0 means nothing qualifies as "strong distance"; only term overlap saves.
    out = apply_relevance_gate(ITEMS, QUERY, enabled=True, shadow=False, distance_floor=0.0)
    # idx0 (overlap mortgage/rate) keep; idx1 (no anchor) drop; idx2 (overlap rate) keep;
    # idx3 (no overlap, dist 0.10 but floor 0.0 so not strong) → drop.
    assert out.dropped_indices == (1, 3)
    assert out.kept_indices == (0, 2)
