"""Deterministic recall relevance gate.

A cheap, LLM-free, embedding-free precision filter applied AFTER the vector/
hybrid search returns. Adapted from wfr-finley's `recall-relevance-gate.ts`
(a SOC2 OpenClaw deployment), but made deliberately CONSERVATIVE for this
homelab, whose recall is explicitly valued for working "on concepts, not exact
text".

The risk of a naive "must share a word with the query" gate is that it discards
exactly the strong *semantic* matches that have no lexical overlap — the whole
point of embeddings. So this gate only ever trims the WEAK tail:

  - A result with vector distance <= `distance_floor` (a strong semantic match)
    is ALWAYS kept, regardless of term overlap.
  - A result above the floor is kept only if it shares >= 1 salient whole-word
    term with the query; otherwise it is a weak match with no lexical anchor and
    is dropped (or, in shadow mode, kept but recorded).

Term matching is by tokenization, not substring: both query and record text are
split into Unicode word tokens, so "rich" never matches "enriched" and "José"
matches correctly. This sidesteps the substring-match class of bug entirely
(cf. wfr-finley FIN-582, where a substring identity match let "Ben" match
"Kevin Bennett").

Everything here is pure and unit-tested in tests/test_relevance_gate.py. The
gate ships DEFAULT-OFF behind `recall_relevance_gate_enabled`; `..._shadow`
logs what WOULD be dropped without dropping it, so the operator can evaluate
real behavior before enabling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Tokenizer: Unicode word runs. `\w` under Python 3 is Unicode-aware, so
# accented letters are included; we lowercase and length-filter afterwards.
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Minimum token length to count as "salient". Drops "a", "to", "of" noise that
# the stopword set might miss and keeps the overlap signal meaningful.
MIN_TERM_LEN = 3

# Closed stopword set — common English function words plus a few conversational
# fillers. Intentionally small and explicit (not a giant NLTK list) so the gate
# stays predictable and the same input always yields the same terms.
STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "are", "was", "were", "has", "have", "had", "his",
        "her", "him", "she", "they", "them", "their", "this", "that", "these",
        "those", "with", "from", "what", "when", "where", "which", "who", "whom",
        "how", "why", "did", "does", "doing", "done", "you", "your", "yours",
        "our", "ours", "its", "out", "off", "not", "but", "can", "could",
        "would", "should", "will", "shall", "may", "might", "must", "about",
        "into", "over", "under", "then", "than", "there", "here", "been",
        "being", "any", "all", "some", "such", "only", "also", "very", "just",
        "get", "got", "let", "say", "said", "tell", "told", "ask", "asked",
        "want", "need", "know", "knew", "like", "make", "made", "use", "used",
        "one", "two", "now", "new", "old", "yes", "way", "day", "thing", "stuff",
    }
)


def extract_salient_terms(text: str | None) -> set[str]:
    """Lowercased, stopword-filtered, length-filtered word-token set of `text`."""
    if not text:
        return set()
    return {
        tok
        for raw in _WORD_RE.findall(text.lower())
        if len(raw) >= MIN_TERM_LEN and (tok := raw) not in STOP_WORDS
    }


def record_references_any_term(record_text: str | None, query_terms: set[str]) -> bool:
    """True if `record_text` shares >= 1 salient term with `query_terms`.

    Whole-word by construction (both sides are tokenized), so there is no
    substring leakage.
    """
    if not query_terms:
        # No salient terms in the query (e.g. all stopwords) → nothing to anchor
        # on, so the gate cannot meaningfully judge relevance. Treat as a match
        # (fail-open) — never let an unanchored query silently empty recall.
        return True
    return bool(extract_salient_terms(record_text) & query_terms)


@dataclass(frozen=True)
class GateDecision:
    """One result's gate outcome. `kept` is what the caller acts on; the rest is
    for shadow logging / metrics."""

    kept: bool
    dropped: bool  # would-be-dropped (in shadow mode, kept stays True but dropped=True)
    reason: str  # "strong-distance" | "term-overlap" | "no-anchor" | "gate-off" | "no-query-terms"


def evaluate_result(
    record_text: str | None,
    dist: float,
    query_terms: set[str],
    *,
    distance_floor: float,
) -> GateDecision:
    """Decide one result. Pure; no I/O.

    - dist <= distance_floor      → keep (strong semantic match), reason strong-distance
    - shares a salient term       → keep, reason term-overlap
    - otherwise                   → drop, reason no-anchor
    """
    if not query_terms:
        return GateDecision(kept=True, dropped=False, reason="no-query-terms")
    if dist <= distance_floor:
        return GateDecision(kept=True, dropped=False, reason="strong-distance")
    if record_references_any_term(record_text, query_terms):
        return GateDecision(kept=True, dropped=False, reason="term-overlap")
    return GateDecision(kept=False, dropped=True, reason="no-anchor")


@dataclass(frozen=True)
class GateOutcome:
    """Aggregate result of applying the gate to a result list."""

    kept_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...]
    shadow: bool

    @property
    def dropped_count(self) -> int:
        return len(self.dropped_indices)


def apply_relevance_gate(
    items: list[tuple[str | None, float]],
    query: str,
    *,
    enabled: bool,
    shadow: bool,
    distance_floor: float,
) -> GateOutcome:
    """Apply the gate to a list of (record_text, dist) tuples.

    Returns the indices to keep and the indices that were (or would be) dropped.
    The caller maps `kept_indices` back onto its own result objects — keeping
    this function free of the server's MemoryRecord type so it is trivially
    unit-testable.

    Behavior:
      - enabled=False              → keep everything (gate is a no-op).
      - enabled=True, shadow=True  → keep everything, but report what WOULD drop.
      - enabled=True, shadow=False → actually drop the no-anchor weak tail.
    """
    if not enabled:
        return GateOutcome(
            kept_indices=tuple(range(len(items))),
            dropped_indices=(),
            shadow=shadow,
        )

    query_terms = extract_salient_terms(query)
    kept: list[int] = []
    dropped: list[int] = []
    for idx, (text, dist) in enumerate(items):
        decision = evaluate_result(text, dist, query_terms, distance_floor=distance_floor)
        if decision.dropped:
            dropped.append(idx)
            if shadow:
                kept.append(idx)  # shadow: record the drop but still return it
        else:
            kept.append(idx)
    # Preserve original order in kept (shadow appends out of order above only
    # when a dropped item is re-kept; re-sort to keep the search ordering).
    kept.sort()
    return GateOutcome(
        kept_indices=tuple(kept),
        dropped_indices=tuple(dropped),
        shadow=shadow,
    )
