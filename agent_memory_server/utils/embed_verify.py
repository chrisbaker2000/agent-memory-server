"""Sample-based cosine probe that a batched embed preserved input order (C2).

Ported from wfr-memory-commons ``storage/embed_verify.py`` (FIN-19 / FIN-295).

``add_memories`` batch-embeds (``aembed_documents([m.text for m in memories])``)
and then ``zip(memories, embeddings, strict=True)``. The strict zip catches a
length mismatch — but NOT a *reordering*: a provider (or LiteLLM, or Ollama)
that silently returned the batch vectors in a different order than the input
texts would write the wrong vector onto the wrong record with no length signal,
producing systematic cross-contamination that is invisible at the call boundary
and corrupts recall for every affected record.

A sample-based cosine probe catches reordering cheaply: re-embed up to
``sample_size`` of the input texts individually and compare each against the
batched vector at the same index. Unrelated texts have cosine ~0; the floor
(0.999 default) tolerates provider-side float rounding without missing a reorder.

**Homelab adaptations** vs. the Finley original:
- This fork has no bulk re-embed / migration path (the provenance migration
  reuses stored vectors; ``rebuild-index`` only recreates the schema). The one
  real batch-embed surface is the live ``add_memories`` write, so the probe is
  config-gated there (``settings.memory_verify_embed_order``, default OFF — the
  hot path does not pay the extra embeds unless an operator opts in), and only
  runs for genuine multi-record batches (a batch of one cannot be reordered).
- The caller (a write path) must **fail-closed** on a *confirmed* reorder — a
  corrupted batch must never be persisted — but **fail-open** on a transient
  inability to verify. So this module raises :class:`EmbeddingOrderError`
  ONLY on a confirmed cosine-floor breach, and returns ``False`` (best-effort
  unverified) when the re-embed itself fails or lengths disagree.
- ``embed_single`` MUST re-embed via the **document** task path
  (``aembed_documents([text])[0]``), matching the batched call — nomic applies a
  different prefix to query vs document text, so a query-path re-embed would
  compare a query vector against a document vector and false-positive below the
  floor (the FIN-632 prefix caveat).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable


logger = logging.getLogger(__name__)


# Records re-embedded per batch. 3 is the smallest value that catches the most
# common reordering bugs (full reverse, rotated-by-one) on a typical batch while
# keeping the extra round-trips low.
_VERIFY_SAMPLE_SIZE_DEFAULT: int = 3

# Minimum cosine similarity between a single-input re-embed and the batched
# vector for the two to be considered the same embedding. 0.999 catches
# reordering (cosine ~0 for unrelated texts) while tolerating float rounding.
_VERIFY_COS_SIM_FLOOR_DEFAULT: float = 0.999


class EmbeddingOrderError(Exception):
    """Raised on a CONFIRMED batch reordering (a sample below the cosine floor).

    Distinct from the best-effort ``False`` return (couldn't verify): this is a
    high-confidence corruption signal, so a write caller must fail-closed and
    NOT persist the batch.

    :ivar sample_index: The first batch position whose re-embed mismatched.
    :ivar cosine_sim: The measured cosine similarity at that position.
    """

    def __init__(self, sample_index: int, cosine_sim: float, log_context: str) -> None:
        self.sample_index = sample_index
        self.cosine_sim = cosine_sim
        super().__init__(
            f"{log_context} embedding-order MISMATCH at index {sample_index} "
            f"(cosine_sim={cosine_sim:.6f}); the embeddings provider may have "
            f"silently reordered the batch result — refusing to persist"
        )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity for embedding-order verification.

    Returns 0.0 if either input is empty, length-mismatched, or has zero norm.
    Numpy is intentionally not imported — a sample-size-3 probe does not need the
    vectorized path and the explicit loop is easier to audit.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


async def verify_embedding_order(
    texts: list[str],
    vectors: list[list[float]],
    *,
    embed_single: Callable[[str], Awaitable[list[float]]],
    sample_size: int = _VERIFY_SAMPLE_SIZE_DEFAULT,
    cos_sim_floor: float = _VERIFY_COS_SIM_FLOOR_DEFAULT,
    log_context: str = "",
) -> bool:
    """Sample-based check that ``vectors[i]`` matches ``embed(texts[i])``.

    Re-embeds up to ``sample_size`` leading texts individually (via the
    caller-supplied ``embed_single`` — which MUST use the same task/prefix path
    as the batched call, see module doc) and compares each against the batched
    vector at the same index.

    :param texts: Input texts passed to the batched embed call, in order.
    :param vectors: Batched embed result, indexed parallel to ``texts``.
    :param embed_single: Async function returning a single embedding for one
        text, via the SAME (document) task path as the batched call.
    :param sample_size: Maximum number of leading positions to verify.
    :param cos_sim_floor: Minimum cosine similarity to accept a sample.
    :param log_context: Free-form tag prefixed on logs (e.g. ``"[add_memories]"``).
    :returns: ``True`` if every sampled position passes (incl. the trivial
        empty / single-record case). ``False`` (best-effort unverified) if
        ``embed_single`` raises or lengths disagree.
    :raises EmbeddingOrderError: on a CONFIRMED reorder (a sample's cosine is
        below ``cos_sim_floor``) — the caller must fail-closed.
    """
    if not texts:
        return True
    if len(texts) != len(vectors):
        logger.error(
            f"{log_context} verify_embedding_order length mismatch: "
            f"texts={len(texts)} vectors={len(vectors)}"
        )
        return False
    count = min(sample_size, len(texts))
    for i in range(count):
        try:
            single_vec = await embed_single(texts[i])
        except Exception as exc:  # best-effort: a re-embed failure is unverified
            logger.error(
                f"{log_context} verify_embedding_order embed_single raised at "
                f"index {i}: {exc}"
            )
            return False
        sim = cosine_similarity(single_vec, vectors[i])
        if sim < cos_sim_floor:
            logger.error(
                f"{log_context} verify_embedding_order MISMATCH at index {i}: "
                f"cosine_sim={sim:.6f} < floor={cos_sim_floor}"
            )
            raise EmbeddingOrderError(i, sim, log_context)
    return True
