"""Regression tests for open item 51 ("read-after-write race").

Root cause (measured live 2026-10-08) was NOT a visibility lag: a just-written
record is visible to FT.SEARCH immediately (RediSearch 2.10, no WORKERS, so HNSW
inserts are synchronous). Two compounding search defects made a fresh record
"invisible" to an unscoped recall:

1. The KNN leg ran at RediSearch's default HNSW EF_RUNTIME (10). On the live
   index, querying a record's own stored vector missed it 2.3% of the time at
   K=15 (6.7% for freshly written outliers), 0% at EF_RUNTIME=100.
2. The BM25 leg escaped punctuation inside a query word (``id=smoke-1``) instead
   of splitting on it like the indexer does, so under AND the text leg returned
   nothing for such queries and could not rescue the KNN miss.

No Redis required: the index is mocked and the query objects handed to it are
inspected.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from agent_memory_server.config import Settings, settings
from agent_memory_server.memory_vector_db import (
    REDISEARCH_DEFAULT_SEPARATORS,
    RedisVLMemoryVectorDatabase,
)
from tests.conftest import MockEmbeddings


def _make_db() -> RedisVLMemoryVectorDatabase:
    mock_index = MagicMock()
    mock_index.exists = AsyncMock(return_value=True)
    mock_index.query = AsyncMock(return_value=[])
    return RedisVLMemoryVectorDatabase(mock_index, MockEmbeddings())


def _queries_sent(db: RedisVLMemoryVectorDatabase) -> list:
    return [call.args[0] for call in db._index.query.call_args_list]


class TestTextQueryTokenizationParity:
    """The text leg must split query words exactly where the indexer splits text."""

    escape = staticmethod(RedisVLMemoryVectorDatabase._escape_text_query)

    def test_smoke_probe_query_splits_key_value_and_hyphens(self):
        # The exact shape of the tests/memory/smoke.sh read-back query.
        assert (
            self.escape("Phase 6 smoke probe id=smoke-1791292422-31922")
            == "Phase 6 smoke probe id smoke 1791292422 31922"
        )

    def test_hyphenated_name_becomes_two_terms(self):
        assert self.escape("Baker-Grant family") == "Baker Grant family"

    def test_email_splits_on_at_and_dot(self):
        assert self.escape("mail chris@example.com") == "mail chris example com"

    def test_url_splits_on_scheme_and_path(self):
        assert (
            self.escape("see https://example.com/a/b?x=1")
            == "see https example com a b x 1"
        )

    @pytest.mark.parametrize("sep", list(REDISEARCH_DEFAULT_SEPARATORS))
    def test_every_default_separator_splits(self, sep):
        assert self.escape(f"alpha{sep}beta") == "alpha beta"

    def test_only_separators_yields_empty_query(self):
        assert self.escape("--- !!! ==") == ""

    def test_underscore_is_not_a_separator(self):
        # RediSearch keeps snake_case as one term; so must the query. The escaper
        # may backslash-escape it, but it must stay a single term.
        result = self.escape("open_claw gateway")
        assert result.split(" ")[1] == "gateway"
        assert len(result.split(" ")) == 2
        assert "open" in result and "claw" in result

    def test_no_escaped_separator_survives(self):
        # Any surviving backslash-escaped separator is a term that can never
        # match indexed text.
        result = self.escape("a-b c=d e.f g@h i/j k:l")
        for sep in REDISEARCH_DEFAULT_SEPARATORS:
            assert f"\\{sep}" not in result

    @pytest.mark.asyncio
    async def test_text_leg_query_uses_split_terms(self):
        db = _make_db()
        await db._text_search(
            "Phase 6 smoke probe id=smoke-1-2", redis_filter=None, limit=15
        )
        (fq,) = _queries_sent(db)
        assert "@text:(Phase 6 smoke probe id smoke 1 2)" in str(fq)


class TestVectorLegEfRuntime:
    """The KNN leg must not run at RediSearch's default EF_RUNTIME."""

    def test_default_is_100(self):
        assert Settings().vector_search_ef_runtime == 100

    @pytest.mark.parametrize("bad", [0, -5])
    def test_non_positive_rejected_at_startup(self, bad):
        with pytest.raises(ValidationError):
            Settings(vector_search_ef_runtime=bad)

    def test_none_allowed_to_restore_redisearch_default(self):
        assert Settings(vector_search_ef_runtime=None).vector_search_ef_runtime is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("hybrid", [True, False])
    async def test_knn_query_carries_configured_ef_runtime(self, hybrid):
        db = _make_db()
        with patch.object(settings, "vector_search_ef_runtime", 100):
            await db.search_memories(query="probe", hybrid_search=hybrid, limit=5)
        vq = _queries_sent(db)[0]
        assert "EF_RUNTIME $EF" in vq.query_string()
        assert vq.params["EF"] == 100

    @pytest.mark.asyncio
    async def test_unset_ef_runtime_omits_clause(self):
        db = _make_db()
        with patch.object(settings, "vector_search_ef_runtime", None):
            await db.search_memories(query="probe", hybrid_search=False, limit=5)
        vq = _queries_sent(db)[0]
        assert "EF_RUNTIME" not in vq.query_string()
        assert "EF" not in vq.params

    @pytest.mark.asyncio
    async def test_range_query_unaffected(self):
        # distance_threshold uses a RangeQuery (EPSILON, not EF_RUNTIME).
        db = _make_db()
        with patch.object(settings, "vector_search_ef_runtime", 100):
            await db.search_memories(
                query="probe", hybrid_search=False, limit=5, distance_threshold=0.3
            )
        vq = _queries_sent(db)[0]
        assert "EF_RUNTIME" not in vq.query_string()

    @pytest.mark.asyncio
    async def test_server_side_recency_knn_carries_ef_runtime(self):
        # SearchRequest.server_side_recency routes to _search_with_recency_aggregation,
        # which builds its own VectorQuery — it must use the same EF_RUNTIME.
        from agent_memory_server.utils.redis_query import RecencyAggregationQuery

        db = _make_db()
        db._index.aggregate = AsyncMock(return_value=[])
        captured = []
        real = RecencyAggregationQuery.from_vector_query.__func__

        def spy(cls, vq, **kwargs):
            captured.append(vq)
            return real(cls, vq, **kwargs)

        with (
            patch.object(settings, "vector_search_ef_runtime", 100),
            patch.object(
                RecencyAggregationQuery, "from_vector_query", classmethod(spy)
            ),
        ):
            await db.search_memories(query="probe", server_side_recency=True, limit=5)
        assert db._index.aggregate.await_count == 1
        (vq,) = captured
        assert "EF_RUNTIME $EF" in vq.query_string()
        assert vq.params["EF"] == 100
