"""Tests for embedding-order verification (C2).

Ported (homelab-adapted) from wfr-memory-commons storage/embed_verify.py.
Covers the pure cosine helper, the verifier's pass / best-effort-False /
fail-closed-raise contract, and the add_memories wiring (gated, batch-only,
document-path re-embed, fail-closed on confirmed reorder). No Redis required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_server.config import settings
from agent_memory_server.memory_vector_db import RedisVLMemoryVectorDatabase
from agent_memory_server.models import MemoryRecord
from agent_memory_server.utils.embed_verify import (
    EmbeddingOrderError,
    cosine_similarity,
    verify_embedding_order,
)
from tests.conftest import MockEmbeddings


# --- cosine_similarity -----------------------------------------------------


def test_cosine_identical_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_scaled_vectors_is_one():
    # cosine is scale-invariant
    assert cosine_similarity([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)


def test_cosine_empty_or_mismatched_or_zero_norm():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0], []) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0  # length mismatch
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero norm


# --- verify_embedding_order: pass paths ------------------------------------


@pytest.mark.asyncio
async def test_verify_empty_texts_trivially_true():
    embed = AsyncMock()
    assert await verify_embedding_order([], [], embed_single=embed) is True
    embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_correct_order_passes():
    texts = ["a", "b", "c"]
    vecs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    # embed_single returns the matching vector for each text
    lut = dict(zip(texts, vecs, strict=True))
    embed = AsyncMock(side_effect=lambda t: lut[t])
    assert await verify_embedding_order(texts, vecs, embed_single=embed) is True


@pytest.mark.asyncio
async def test_verify_samples_only_sample_size():
    texts = ["a", "b", "c", "d", "e"]
    vecs = [[1.0, 0.0]] * 5
    embed = AsyncMock(return_value=[1.0, 0.0])
    await verify_embedding_order(texts, vecs, embed_single=embed, sample_size=2)
    assert embed.await_count == 2  # only the leading sample_size positions


# --- verify_embedding_order: fail-closed (confirmed reorder) ---------------


@pytest.mark.asyncio
async def test_verify_reorder_raises_mismatch():
    # Batch returned reversed: vectors[0] belongs to texts[2], etc.
    texts = ["a", "b", "c"]
    true_vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    batched_reversed = [true_vecs["c"], true_vecs["b"], true_vecs["a"]]
    embed = AsyncMock(side_effect=lambda t: true_vecs[t])
    with pytest.raises(EmbeddingOrderError) as ei:
        await verify_embedding_order(texts, batched_reversed, embed_single=embed)
    assert ei.value.sample_index == 0  # texts[0]=a vs vec for c → mismatch


@pytest.mark.asyncio
async def test_verify_rotated_by_one_raises():
    texts = ["a", "b", "c"]
    true_vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [2.0, 3.0]}
    rotated = [true_vecs["b"], true_vecs["c"], true_vecs["a"]]
    embed = AsyncMock(side_effect=lambda t: true_vecs[t])
    with pytest.raises(EmbeddingOrderError):
        await verify_embedding_order(texts, rotated, embed_single=embed)


@pytest.mark.asyncio
async def test_verify_floor_tolerates_float_rounding():
    # A vector that differs only by tiny rounding stays above the 0.999 floor.
    texts = ["a"]
    batched = [[1.0, 2.0, 3.0]]
    embed = AsyncMock(return_value=[1.0000001, 2.0000001, 2.9999999])
    assert await verify_embedding_order(texts, batched, embed_single=embed) is True


# --- verify_embedding_order: best-effort False (unverified) ----------------


@pytest.mark.asyncio
async def test_verify_length_mismatch_returns_false():
    embed = AsyncMock()
    # 2 texts, 1 vector — cannot sample-verify
    assert (
        await verify_embedding_order(["a", "b"], [[1.0]], embed_single=embed) is False
    )
    embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_embed_single_failure_returns_false():
    embed = AsyncMock(side_effect=RuntimeError("ollama down"))
    # Infra failure is unverified (False), NOT a mismatch (no raise).
    assert (
        await verify_embedding_order(["a"], [[1.0, 0.0]], embed_single=embed) is False
    )


# --- add_memories wiring ---------------------------------------------------


def _db():
    return RedisVLMemoryVectorDatabase(MagicMock(), MockEmbeddings())


@pytest.fixture
def _verify_on(monkeypatch):
    monkeypatch.setattr(settings, "memory_verify_embed_order", True)


def _records(texts):
    return [MemoryRecord(id=f"m{i}", text=t) for i, t in enumerate(texts)]


async def _run_add(db, memories):
    """Drive add_memories with a mocked index.load (no Redis)."""
    db._index = MagicMock()
    db._index.load = AsyncMock(return_value=[m.id for m in memories])
    with patch.object(db, "_ensure_index", AsyncMock()):
        return await db.add_memories(memories)


@pytest.mark.asyncio
async def test_add_memories_verify_off_skips_check(monkeypatch):
    monkeypatch.setattr(settings, "memory_verify_embed_order", False)
    db = _db()
    # Batch returns reversed vectors, but verification is OFF → no raise.
    true_vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    db.embeddings.aembed_documents = AsyncMock(
        return_value=[true_vecs["b"], true_vecs["a"]]
    )
    ids = await _run_add(db, _records(["a", "b"]))
    assert ids == ["m0", "m1"]


@pytest.mark.asyncio
async def test_add_memories_single_record_not_checked(_verify_on):
    db = _db()
    # Single record: a batch of one cannot be reordered → embed called once
    # (the batch), never re-embedded for verification.
    db.embeddings.aembed_documents = AsyncMock(return_value=[[1.0, 0.0]])
    ids = await _run_add(db, _records(["solo"]))
    assert ids == ["m0"]
    assert db.embeddings.aembed_documents.await_count == 1  # batch only


@pytest.mark.asyncio
async def test_add_memories_correct_order_persists(_verify_on):
    db = _db()
    true_vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}

    async def fake_embed(texts):
        return [true_vecs[t] for t in texts]

    db.embeddings.aembed_documents = AsyncMock(side_effect=fake_embed)
    ids = await _run_add(db, _records(["a", "b"]))
    assert ids == ["m0", "m1"]
    db._index.load.assert_awaited_once()  # persisted


@pytest.mark.asyncio
async def test_add_memories_reorder_fails_closed(_verify_on):
    db = _db()
    true_vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}

    async def fake_embed(texts):
        # batch of 2 → return reversed; single re-embed → return true vector
        if len(texts) > 1:
            return [true_vecs[texts[1]], true_vecs[texts[0]]]
        return [true_vecs[texts[0]]]

    db.embeddings.aembed_documents = AsyncMock(side_effect=fake_embed)
    with pytest.raises(EmbeddingOrderError):
        await _run_add(db, _records(["a", "b"]))
    # Corrupted batch must NOT be persisted.
    db._index.load.assert_not_awaited()
