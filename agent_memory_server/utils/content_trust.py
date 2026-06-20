"""Reference-record write-protection — trust-rank mutation guard (C1).

Ported from wfr-memory-commons ``utils/content_trust.py`` (FIN-466 / M-4), the
commons analog of Anthropic's memory-stores ``read_only`` mount: a lower-trust
writer must never be able to **supersede or delete** a higher-trust (canonical /
identity / reference) record. Without this, a poisoned or runaway agent path —
the highest-fan-out indirect-prompt-injection surface in the system — could
destroy or overwrite an operator-authored canonical fact (a family identity, an
infra fact) and there is no attribution-chain-independent gate to stop it.

**Single-tenant collapse.** Finley keys trust off a per-request, bearer-token
``ConsumerId`` and a three-tier ``TrustLevel`` (system / first_party / agent).
The homelab is single-tenant with ``auth_mode=disabled`` (every caller is the
same ``current_user`` on loopback), so there is no per-consumer identity to map.
We collapse the model to two tiers — :class:`TrustLevel.OPERATOR` (Finley's
system + first_party: operator/system writers, canonical, not exposed to
untrusted input) and :class:`TrustLevel.AGENT` (Finley's agent/extraction: the
LLM-driven write path that processes untrusted tool output / web content /
inbound messages) — and derive the tier from a single server-side,
agent-unforgeable signal: presence of a valid operator shared-secret
(``X-Operator-Token``) on the mutating request. The agent path does not hold
that secret, so it cannot mint an OPERATOR-tier write or escape the gate.

**Dormant by default.** When no operator token is configured
(``settings.memory_operator_token is None``) no write is ever stamped OPERATOR,
so :func:`record_trust_level` resolves every record (including all legacy
records) to :data:`LOWEST_TRUST_LEVEL` and :func:`is_reference_protected_mutation`
can never fire — a pure no-op until an operator deliberately opts in. This keeps
the port safe to ship enabled and reversible by un-setting one env var.

Defense-in-depth, not the only line: the gateway already gates the agent's
``memory_forget`` tool via the ``safety-contract`` plugin, and ``supersede`` is
not exposed to the agent as a tool at all. This module is the **server-side**
backstop that survives a gateway bypass, a disabled safety rule, or a future
tool, and it keys on a signal the agent cannot forge.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class TrustLevel(StrEnum):
    """Coarse author-trust tier stamped on a stored record.

    Ordered least- to most-trusted. The value reflects how exposed the *writing
    path* is to untrusted external input — NOT the sensitivity of the content.

    - :attr:`AGENT`: the LLM-driven write path (``memory_store`` tool,
      extraction, working-memory promotion). It processes untrusted tool output,
      web/search results, and inbound channel messages, so anything it authors
      is lowest-trust by default. This is also the fail-safe value for any
      unclassified or legacy record.
    - :attr:`OPERATOR`: operator/system writers (a human operator via a script
      or manual call carrying the operator token, migrations). Canonical; not
      exposed to raw untrusted input.
    """

    AGENT = "agent"
    OPERATOR = "operator"


# Least-trusted tier — the fail-safe value. Used both as the floor for any
# record that carries no (or an unrecognized) ``trust_level`` and as the default
# tier for a caller that presents no valid operator token. Fail-safe-to-most-
# restrictive: an unclassifiable writer is never trusted above the agent tier,
# and an unclassifiable record is never *treated as protected* (so the gate
# can't over-block legacy records — see :func:`record_trust_level`).
LOWEST_TRUST_LEVEL: TrustLevel = TrustLevel.AGENT


# Trust ordering for the mutation gate — most-privileged highest. A caller may
# mutate a record only when its own tier ranks at or above the record's. Kept as
# an explicit table (not ``IntEnum`` ordinals) so a future third tier can't
# silently inherit a rank; ``test_content_trust`` asserts it is total over
# ``TrustLevel``.
_TRUST_RANK: dict[TrustLevel, int] = {
    TrustLevel.AGENT: 0,
    TrustLevel.OPERATOR: 1,
}


def trust_rank(level: TrustLevel) -> int:
    """Return the integer privilege rank of a trust tier (higher = more trusted).

    :param level: The :class:`TrustLevel` to rank.
    :returns: ``0`` (agent) ≤ ``1`` (operator).
    """
    return _TRUST_RANK[level]


def derive_trust_level(*, is_operator: bool) -> TrustLevel:
    """Map the server-resolved operator signal to an author-trust tier.

    The single-tenant collapse of Finley's per-consumer trust table: the only
    server-side, agent-unforgeable signal is whether the mutating request
    carried a valid operator token. Never accepted from client-controlled body
    fields — only from the token check in :func:`is_operator_token`.

    :param is_operator: True iff the caller presented a valid operator token.
    :returns: :attr:`TrustLevel.OPERATOR` when ``is_operator``, else
        :data:`LOWEST_TRUST_LEVEL`.
    """
    return TrustLevel.OPERATOR if is_operator else LOWEST_TRUST_LEVEL


def is_operator_token(presented: str | None, configured: str | None) -> bool:
    """Constant-time check of a presented operator token against the configured one.

    Fail-safe-to-untrusted: returns False when no token is configured
    (the dormant default — nobody is ever OPERATOR) or when either value is
    missing/empty. Uses :func:`hmac.compare_digest` so a timing side-channel
    cannot be used to recover the secret a character at a time.

    :param presented: The ``X-Operator-Token`` header value from the request
        (``None`` when absent).
    :param configured: ``settings.memory_operator_token`` (``None`` when unset).
    :returns: True iff a non-empty token is configured and matches exactly.
    """
    if not configured or not presented:
        return False
    return hmac.compare_digest(presented, configured)


def parse_trust_level(value: Any) -> TrustLevel:
    """Coerce a stored ``trust_level`` value to a :class:`TrustLevel`.

    Fail-safe to :data:`LOWEST_TRUST_LEVEL` for any value that is missing,
    not a recognized tier, or the wrong type — an unclassifiable record is
    never treated as more trusted than the lowest tier.

    :param value: The raw value read off a stored record's ``trust_level``.
    :returns: The matching :class:`TrustLevel`, or :data:`LOWEST_TRUST_LEVEL`.
    """
    if isinstance(value, TrustLevel):
        return value
    if isinstance(value, str):
        try:
            return TrustLevel(value)
        except ValueError:
            return LOWEST_TRUST_LEVEL
    return LOWEST_TRUST_LEVEL


def record_trust_level(record: Mapping[str, Any] | Any) -> TrustLevel:
    """Return the stored author-trust tier of a record, fail-safe to lowest.

    Accepts either a mapping (a stored hash dict) or any object exposing a
    ``trust_level`` attribute (a :class:`~agent_memory_server.models.MemoryRecord`).
    Legacy records written before this field existed carry no ``trust_level`` and
    resolve to :data:`LOWEST_TRUST_LEVEL` — so the reference gate never
    over-blocks a legacy (effectively agent-tier) record, while every record an
    operator stamped OPERATOR is protected.

    :param record: A stored record dict or a record object.
    :returns: The record's :class:`TrustLevel`.
    """
    if isinstance(record, Mapping):
        raw = record.get("trust_level")
    else:
        raw = getattr(record, "trust_level", None)
    return parse_trust_level(raw)


def is_reference_protected_mutation(
    *, caller: TrustLevel, record: Mapping[str, Any] | Any
) -> bool:
    """Return True iff ``caller`` must be BLOCKED from mutating ``record``.

    The reference-record write-protection rule: a mutation (supersede / delete)
    is rejected when the caller's tier ranks strictly below the target record's
    tier. Equal-or-higher callers are allowed — a record's own tier (and any
    higher-trust principal) may still manage it.

    Concretely with the two-tier model: an ``agent``-tier caller cannot
    supersede/delete an ``operator``-tier (canonical) record; an ``operator``
    caller outranks everything; ``agent`` managing an ``agent``-tier record is
    unaffected (equal rank). Both sides are server-derived — the record tier was
    stamped server-side at write, the caller tier from the operator-token check —
    so the gate cannot be escaped from inside a compromised agent path.

    :param caller: The server-resolved calling tier.
    :param record: The target record being mutated (dict or record object).
    :returns: True when the mutation is reference-protected (must 403 / forbid).
    """
    return trust_rank(caller) < trust_rank(record_trust_level(record))


class ReferenceProtectedError(Exception):
    """Raised when a lower-trust caller attempts to mutate a protected record.

    Carries the IDs that were blocked so the API layer can return an actionable
    403. No record is mutated when this is raised (atomic refuse).

    :ivar blocked_ids: The target record IDs the caller is not allowed to mutate.
    :ivar caller: The caller's resolved trust tier value.
    """

    def __init__(self, blocked_ids: list[str], caller: TrustLevel) -> None:
        self.blocked_ids = blocked_ids
        self.caller = caller.value
        super().__init__(
            f"Reference-protected: caller tier '{caller.value}' may not mutate "
            f"higher-trust record(s): {blocked_ids}"
        )
