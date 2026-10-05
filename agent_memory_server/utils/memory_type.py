"""Normalization of LLM-emitted ``memory_type`` values for extracted memories.

The extraction LLM occasionally emits a ``type`` outside the ``MemoryTypeEnum``
(observed live: ``"epistemic"`` — a near-miss of ``"episodic"``, plausibly primed
by the discrete prompt's separate *epistemic* ``kind`` field). Passed straight into
``MemoryRecord`` that raises a pydantic ``ValidationError``; historically this
aborted the whole extraction batch (LAB-487, LAB-533).

This module is deliberately dependency-light (models + telemetry + logging only)
so both ``extraction.py`` and ``memory_strategies.py`` can import it without an
import cycle. ``extraction.py`` re-exports the public names for back-compat.

Policy (LAB-533) — never widen the enum; normalize BEFORE validation:

* ``"episodic"`` / ``"semantic"`` (case/whitespace-insensitive) → canonical, silent.
* A known alias/typo (``_MEMORY_TYPE_ALIASES``, e.g. ``"epistemic"`` → ``"episodic"``)
  → its mapped value, WARNING + counter.

The raw emitted value is NEVER logged or used as a metric attribute (it may carry
a secret fragment and bypasses the write-funnel redaction): only a known alias key
is named; anything else is logged as type/length/sha256-prefix metadata.
* Any other non-empty string — including ``"message"``, which is a valid enum value
  but reserved for raw conversation records — → ``"semantic"`` (the safe default:
  a timeless fact is the least-wrong reading of an unclassifiable extraction),
  WARNING + counter.
* A non-string emission (list/dict/number — unhashable values must not raise)
  → ``"semantic"``, WARNING + counter.
* Missing (``None``) or empty → the call-site ``default`` (each site's historical
  default), silent: an absent field is not an emitted value to coerce.
"""

from __future__ import annotations

import hashlib

from agent_memory_server.logging import get_logger
from agent_memory_server.models import MemoryTypeEnum
from agent_memory_server.telemetry import record_counter


logger = get_logger(__name__)

# Valid LLM-emitted memory types for EXTRACTED memories. MemoryTypeEnum also
# includes "message", but that is reserved for raw conversation-message records —
# an extracted/derived fact must never be typed "message" (the session-thread path
# also stamps session_id, so a "message"-typed extracted fact would pollute
# message-only reconstruction/search paths).
VALID_MEMORY_TYPES = frozenset(
    {MemoryTypeEnum.EPISODIC.value, MemoryTypeEnum.SEMANTIC.value}
)

# Known near-miss / cross-field confusions → the type they most plausibly meant.
# Keys are normalized (strip + lower). Keep values inside VALID_MEMORY_TYPES (a
# test locks this). "event"/"fact"/"preference"/"summary" are `kind` values the
# LLM sometimes writes into `type` — map them to the type each kind implies.
_MEMORY_TYPE_ALIASES: dict[str, str] = {
    "epistemic": MemoryTypeEnum.EPISODIC.value,  # observed live (LAB-487/533)
    "episode": MemoryTypeEnum.EPISODIC.value,
    "episodical": MemoryTypeEnum.EPISODIC.value,
    "event": MemoryTypeEnum.EPISODIC.value,
    "semantics": MemoryTypeEnum.SEMANTIC.value,
    "fact": MemoryTypeEnum.SEMANTIC.value,
    "factual": MemoryTypeEnum.SEMANTIC.value,
    "preference": MemoryTypeEnum.SEMANTIC.value,
    "summary": MemoryTypeEnum.SEMANTIC.value,
}

# Fallback for a present-but-unrecognized emission (LAB-533).
UNKNOWN_MEMORY_TYPE_FALLBACK = MemoryTypeEnum.SEMANTIC.value

# The established default for a MISSING type at the strategy-aware construction
# site (historical `new_memory.get("type", "episodic")`).
DEFAULT_MEMORY_TYPE = MemoryTypeEnum.EPISODIC.value

COERCION_METRIC = "memory_server.extraction.memory_type_coerced"


def describe_unsafe_value(value: object) -> str:
    """Non-reversible, log-safe description of an untrusted LLM value.

    Never includes the value or any substring of it — only the Python type name,
    the length (of ``str(value)``) and the first 8 hex chars of its sha256. The
    raw value is junk by definition and may carry a secret/PII fragment that the
    write-funnel redaction (content_security) never sees on this path (codex F1).
    """
    rendered = value if isinstance(value, str) else repr(value)
    digest = hashlib.sha256(rendered.encode("utf-8", "replace")).hexdigest()[:8]
    return f"<{type(value).__name__} len={len(rendered)} sha256={digest}>"


def _report_coercion(
    original: object, coerced: str, reason: str, site: str, label: str | None
) -> None:
    """WARNING + SigNoz counter for one coercion — WITHOUT the raw value.

    ``label`` is a KNOWN, trusted key (an alias-map key or the reserved enum value
    ``"message"``) and is the only form of the original that may appear in the log
    or as a metric attribute. Anything else is described by
    :func:`describe_unsafe_value` in the log and never reaches metric attributes
    (no leakage, bounded cardinality).
    """
    shown = repr(label) if label is not None else describe_unsafe_value(original)
    # f-string, not %-args: the structlog config has no PositionalArgumentsFormatter,
    # so positional args would not be interpolated into the rendered event.
    logger.warning(
        f"Coerced off-enum extracted memory_type {shown}"
        f" -> {coerced!r} (reason={reason}, site={site})"
    )
    attrs = {"reason": reason, "coerced_to": coerced, "site": site}
    if reason == "alias" and label is not None:
        attrs["original"] = label
    record_counter(COERCION_METRIC, attributes=attrs)


def coerce_memory_type(
    value: object,
    default: str = DEFAULT_MEMORY_TYPE,
    *,
    site: str = "unspecified",
) -> str:
    """Coerce an LLM-emitted ``memory_type`` to a valid EXTRACTED memory type.

    Always returns a member of :data:`VALID_MEMORY_TYPES`; never raises (the
    ``isinstance(value, str)`` guard is load-bearing — a bare ``in`` membership
    test raises ``TypeError`` on an unhashable emission like ``["semantic"]``).

    Args:
        value: The raw ``type`` value from the extraction LLM's JSON.
        default: Returned SILENTLY when ``value`` is missing (``None``) or empty —
            each construction site's historical default. Must be a valid type.
        site: Call-site label carried on the WARNING log + counter attributes.

    Returns:
        ``"episodic"`` or ``"semantic"`` (see module docstring for the policy).
    """
    if default not in VALID_MEMORY_TYPES:
        raise ValueError(f"default memory_type {default!r} is not extractable")
    if value is None:
        return default
    if not isinstance(value, str):
        _report_coercion(
            value, UNKNOWN_MEMORY_TYPE_FALLBACK, "invalid_type", site, None
        )
        return UNKNOWN_MEMORY_TYPE_FALLBACK
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in VALID_MEMORY_TYPES:
        return normalized
    alias = _MEMORY_TYPE_ALIASES.get(normalized)
    if alias is not None:
        _report_coercion(value, alias, "alias", site, normalized)
        return alias
    if normalized == MemoryTypeEnum.MESSAGE.value:
        _report_coercion(
            value, UNKNOWN_MEMORY_TYPE_FALLBACK, "reserved", site, normalized
        )
    else:
        _report_coercion(value, UNKNOWN_MEMORY_TYPE_FALLBACK, "unknown", site, None)
    return UNKNOWN_MEMORY_TYPE_FALLBACK


# Shared extraction-prompt constraint (LAB-533). Fully resolved (no template
# placeholders) and brace-free, so it is safe both as a str.format() argument in
# the built-in prompts and inside the custom-strategy preamble.
MEMORY_TYPE_PROMPT_CONSTRAINT = (
    'MEMORY TYPE (STRICT): every memory\'s "type" field MUST be exactly one of '
    + " or ".join(f'"{t}"' for t in sorted(VALID_MEMORY_TYPES))
    + ' (lowercase, no other value). "episodic" = a specific time-anchored '
    'episode/event; "semantic" = a timeless fact, preference, or general '
    'knowledge. Do NOT invent other values (e.g. NOT "epistemic", "message", '
    '"fact", "event"). "type" is a DIFFERENT field from "kind": the epistemic '
    'category ("fact", "event", "preference", "opinion", "belief", "summary") '
    'belongs ONLY in "kind", never in "type".'
)
