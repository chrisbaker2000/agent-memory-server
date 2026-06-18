"""Write-time content security for stored memory.

Ported from wfr-memory-commons (`further-memory`, the SOC2 Further memory
service) — specifically the load-bearing, pure-Python halves of its
`utils/dlp.py` (secret redaction) and `utils/content_trust.py` (Unicode
sanitization + prompt-injection flagging). The multi-tenant machinery from
those modules (per-consumer `ConsumerId` trust tiers, visibility-band
`DataClassification`, per-consumer egress DLP) is deliberately NOT ported: this
homelab is single-tenant (Pat + family), so those concepts collapse to noise.

Why this matters here: every stored memory is read back into an LLM context via
recall, so the memory store is a RAG re-injection surface. Two concrete risks:

  1. A secret (API key, private-key block, token) captured into a memory becomes
     an exfiltration target the moment recall surfaces it. `redact_secrets`
     scrubs known credential shapes to `[REDACTED:<label>]` *before* the text is
     ever embedded or persisted. Redact, don't reject — a stray secret should
     not discard an otherwise useful memory.
  2. Control-char / zero-width / bidi obfuscation can smuggle prompt-injection
     payloads past word-boundary scanners and spoof log output.
     `sanitize_memory_text` NFC-normalizes and strips C0/C1/DEL + format (Cf)
     chars. `scan_for_injection` then flags (never rejects) instruction-shaped
     text so the operator can review it via logs/telemetry.

All three are pure, deterministic, dependency-free (stdlib `re`/`unicodedata`),
and unit-tested in `tests/test_content_security.py`. They are wired into the
universal write funnel `long_term_memory.index_long_term_memories`, so every
write path (API, memory-manager, extraction, working-memory promotion, plugin)
is covered. Behaviour is config-gated (`settings.memory_*`) and defaults to ON
for the cheap removal-only controls (sanitize, redact) — see config.py.

The secret-pattern set mirrors the repo-wide `~/.claude/hooks/scan-for-secrets.sh`
and the original further-memory `_SECRET_PATTERNS`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# ============================================================================
# Unicode sanitization (ported from content_trust.sanitize_memory_text)
# ============================================================================

# Whitespace control chars that are legitimate prose and must survive.
_ALLOWED_CONTROL_CHARS: frozenset[str] = frozenset({"\t", "\n", "\r"})


def _is_strippable_control(ch: str) -> bool:
    """True if `ch` is a control/format char we strip on write."""
    if ch in _ALLOWED_CONTROL_CHARS:
        return False
    codepoint = ord(ch)
    if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
        # C0 controls, DEL, and C1 controls.
        return True
    # Category Cf: zero-width chars, BOM, bidi overrides, soft hyphen. These
    # survive NFC and would otherwise let "ig​nore"-style obfuscation evade
    # the injection scanner's word boundaries (and bidi overrides spoof logs).
    return unicodedata.category(ch) == "Cf"


def sanitize_memory_text(text: str) -> str:
    """Normalize + strip control characters from memory text on write.

    - Unicode NFC normalization folds equivalent code-point sequences to a
      canonical form so search + dedup compare like-for-like.
    - C0 / C1 / DEL control characters are removed except ``\\t`` / ``\\n`` /
      ``\\r`` (legitimate prose whitespace).
    - Format characters (category Cf) are removed.

    Pure function; never raises. Removal-only (plus NFC), so the result never
    exceeds the input length.
    """
    normalized = unicodedata.normalize("NFC", text)
    if not any(_is_strippable_control(ch) for ch in normalized):
        return normalized
    return "".join(ch for ch in normalized if not _is_strippable_control(ch))


# ============================================================================
# Secret redaction (ported from dlp._SECRET_PATTERNS / redact_secrets)
# ============================================================================

# Secret signatures, ordered most- to least-specific so the emitted label is the
# narrowest correct one (e.g. an Anthropic ``sk-ant-`` key must be tried before
# the broader OpenAI ``sk-`` rule, which would otherwise also match it). The
# trailing quantifiers are deliberately greedy: glued-to-adjacent-text keys
# over-redact the trailing run — the SAFE direction (redact-not-reject). Input
# is length-bounded upstream (MAX_MEMORY_OUTPUT_CHARS=1000), so no ReDoS risk.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Private-key blocks first — multi-line, redact the whole block. Non-greedy
    # body keeps two blocks in one document from collapsing into one redaction.
    (
        "private key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    # A lone BEGIN marker (truncated paste) — still a leak signal worth scrubbing.
    ("private key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_-]{90,}")),
    ("OpenAI API key", re.compile(r"sk-[A-Za-z0-9_-]{40,}")),
    ("Stripe secret key", re.compile(r"sk_live_[A-Za-z0-9]{20,}")),
    ("GCP API key", re.compile(r"AIza[A-Za-z0-9_-]{35}")),
    ("AWS access key", re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}")),
    # Fine-grained PATs before classic tokens (most- to least-specific invariant).
    ("GitHub fine-grained PAT", re.compile(r"github_pat_[A-Za-z0-9_]{60,}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("Slack token", re.compile(r"xox[bpasr]-[0-9]+-[0-9]+-[A-Za-z0-9]{20,}")),
)


def redact_secrets(text: str) -> tuple[str, tuple[str, ...]]:
    """Replace known secret/credential shapes in ``text`` with placeholders.

    Each match is replaced by ``[REDACTED:<label>]`` so the surrounding context
    survives while the secret never reaches the embedding model or storage.
    Patterns are applied most- to least-specific so the placeholder carries the
    narrowest correct label.

    Pure function; never raises. Idempotent — the placeholder contains none of
    the secret shapes, so re-running it is a no-op.

    :returns: ``(redacted_text, labels)`` where ``labels`` is the sorted, unique
        tuple of secret-type labels that matched (empty when clean).
    """
    redacted = text
    matched: set[str] = set()
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(redacted):
            matched.add(label)
            redacted = pattern.sub(f"[REDACTED:{label}]", redacted)
    return redacted, tuple(sorted(matched))


# ============================================================================
# Prompt-injection flagging (ported from content_trust.scan_for_injection)
# ============================================================================

# Injection-signal signatures grouped by category. Heuristic, deliberately
# sensitivity-biased (flag-and-keep makes false positives cheap). A category
# fires if ANY of its patterns match. This is a FLAGGING heuristic, not a
# security boundary — the caller logs + emits telemetry, it does not reject the
# record.
_INJECTION_SIGNATURES: dict[str, tuple[re.Pattern[str], ...]] = {
    # Attempts to override the standing instruction hierarchy.
    "instruction_override": (
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,40}?\b"
            r"(?:previous|prior|above|earlier|all|any)\b.{0,30}?\b"
            r"(?:instruction|instructions|prompt|prompts|context|rule|rules|"
            r"directive|directives|message|messages|guardrail|guardrails)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(r"\bnew\s+instructions?\s*[:\-]", re.IGNORECASE),
        re.compile(
            r"\b(?:you\s+are\s+now|from\s+now\s+on|act\s+as|pretend\s+to\s+be|"
            r"developer\s+mode|jailbreak)\b",
            re.IGNORECASE,
        ),
    ),
    # Attempts to forge a higher-trust role/turn boundary or escape a wrapper.
    "wrapper_escape": (
        re.compile(r"</?\s*untrusted-external\s*>", re.IGNORECASE),
        re.compile(r"</?\s*system\s*>", re.IGNORECASE),
        re.compile(r"<\|\s*(?:im_start|im_end|system|endoftext)\s*\|>", re.IGNORECASE),
        re.compile(r"\[/?\s*INST\s*\]", re.IGNORECASE),
        re.compile(r"(?m)^\s*(?:system|assistant)\s*:", re.IGNORECASE),
    ),
    # Tool/function-call-shaped text or directives to invoke a tool.
    "tool_directive": (
        re.compile(
            r"<\s*(?:tool_call|function_call|tool_code|invoke)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:call|invoke|execute|run|use)\b\s+the\b.{0,30}?\btool\b",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r'"\s*(?:name|tool_name|function)\s*"\s*:\s*"[^"]+"\s*,\s*"\s*'
            r'(?:arguments|parameters|args|input)\s*"',
            re.IGNORECASE,
        ),
    ),
    # Imperative destructive / exfiltration commands aimed at the agent.
    "exfiltration_destruction": (
        re.compile(
            r"\b(?:delete|drop|wipe|erase|destroy|purge|truncate)\b.{0,30}?\b"
            r"(?:all|every|database|table|tables|records?|memory|memories|data)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
        re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
        re.compile(
            r"\b(?:exfiltrate|leak|send|forward|email|post|upload|reveal)\b.{0,40}?\b"
            r"(?:secret|secrets|credential|credentials|api[\s_-]?key|api[\s_-]?keys|"
            r"password|passwords|token|tokens|private\s+key)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
}


def scan_for_injection(text: str) -> list[str]:
    """Return the sorted set of injection-signal categories matched in ``text``.

    Heuristic, deterministic detector for content that reads like an imperative
    instruction targeting the agent rather than a stored fact. An empty list
    means no signal. Scan the *sanitised* text so control-char obfuscation can't
    slip a payload past word boundaries.
    """
    matched = {
        category
        for category, patterns in _INJECTION_SIGNATURES.items()
        if any(pattern.search(text) for pattern in patterns)
    }
    return sorted(matched)


# ============================================================================
# Orchestrator
# ============================================================================


@dataclass(frozen=True)
class ContentSecurityResult:
    """Outcome of :func:`apply_content_security` for one record.

    :ivar text: The text to persist + embed — sanitised and (if enabled) with
        secrets redacted out.
    :ivar changed: True when ``text`` differs from the input (sanitise stripped
        chars and/or a secret was redacted) — lets the caller skip a
        ``model_copy`` when nothing changed.
    :ivar secrets_redacted: True when at least one secret was redacted.
    :ivar secret_labels: Matched secret-type labels — for the WARNING log +
        telemetry only, NOT persisted on the record (so the record itself never
        advertises that it once held a credential).
    :ivar injection_signals: Matched injection categories — for the WARNING log
        + telemetry only. Flag-and-keep: never mutates or rejects the record.
    """

    text: str
    changed: bool
    secrets_redacted: bool
    secret_labels: tuple[str, ...]
    injection_signals: tuple[str, ...]


def apply_content_security(
    text: str,
    *,
    sanitize: bool = True,
    redact: bool = True,
    scan: bool = True,
) -> ContentSecurityResult:
    """Run the full write-time content-security policy for one record's text.

    Order: sanitise (NFC + control-char strip) → redact secrets → scan the
    cleaned text for injection signals. Each step is independently gated so the
    operator can disable a control without disabling the others.

    Pure function; never raises. Returns the cleaned text plus flags/labels for
    the caller to log + emit as telemetry.
    """
    original = text
    cleaned = sanitize_memory_text(text) if sanitize else text

    secret_labels: tuple[str, ...] = ()
    if redact:
        cleaned, secret_labels = redact_secrets(cleaned)

    injection_signals: tuple[str, ...] = ()
    if scan:
        injection_signals = tuple(scan_for_injection(cleaned))

    return ContentSecurityResult(
        text=cleaned,
        changed=cleaned != original,
        secrets_redacted=bool(secret_labels),
        secret_labels=secret_labels,
        injection_signals=injection_signals,
    )
