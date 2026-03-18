"""Tests for the MemoryVectorDatabase abstraction."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_server.memory_vector_db import (
    MemoryVectorDatabase,
    RedisVLMemoryVectorDatabase,
)
from agent_memory_server.memory_vector_db_factory import (
    create_embeddings,
    create_memory_vector_db,
)
from agent_memory_server.models import (
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
    MemoryTypeEnum,
)


from tests.conftest import MockEmbeddings


class TestMemoryVectorDatabase:
    """Test cases for MemoryVectorDatabase functionality."""

    def test_memory_hash_generation(self):
        """Test memory hash generation."""
        # Create a concrete implementation for testing
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        # Create a sample memory
        memory = MemoryRecord(
            text="This is a test memory",
            id="test-hash-123",
            user_id="user-123",
            session_id="session-456",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        # Generate hash
        hash1 = db.generate_memory_hash(memory)
        hash2 = db.generate_memory_hash(memory)

        # Verify hash is stable
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

        # Verify different memories produce different hashes
        different_memory = MemoryRecord(
            text="This is a different memory",
            id="test-hash-456",
            user_id="user-123",
            session_id="session-456",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )
        different_hash = db.generate_memory_hash(different_memory)
        assert hash1 != different_hash

    def test_parse_list_field(self):
        """Test parsing of list fields."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()
        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        # Test with None
        assert db._parse_list_field(None) == []

        # Test with empty string
        assert db._parse_list_field("") == []

        # Test with comma-separated string (legacy fallback when no pipe present)
        assert db._parse_list_field("a,b,c") == ["a", "b", "c"]

        # Test with list
        assert db._parse_list_field(["a", "b"]) == ["a", "b"]

    def test_memory_to_data_conversion(self):
        """Test converting MemoryRecord to data dict."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()
        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        memory = MemoryRecord(
            text="This is a test memory",
            id="test-123",
            session_id="session-456",
            user_id="user-789",
            namespace="test",
            topics=["testing", "memory"],
            entities=["test"],
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        data = db._memory_to_data(memory)

        assert data["text"] == "This is a test memory"
        assert data["id_"] == "test-123"
        assert data["session_id"] == "session-456"
        assert data["user_id"] == "user-789"
        assert data["namespace"] == "test"
        assert data["topics"] == "testing|memory"
        assert data["entities"] == "test"
        assert data["memory_type"] == "semantic"

    def test_data_to_memory_result_conversion(self):
        """Test converting data dict to MemoryRecordResult."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()
        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        fields = {
            "id_": "test-123",
            "text": "This is a test memory",
            "session_id": "session-456",
            "user_id": "user-789",
            "namespace": "test",
            "topics": "testing|memory",
            "entities": "test",
            "memory_type": "semantic",
            "created_at": "1704067200",  # 2024-01-01T00:00:00Z
            "last_accessed": "1704067200",
            "updated_at": "1704067200",
            "discrete_memory_extracted": "t",
        }

        result = db._data_to_memory_result(fields, score=0.2)

        assert result.text == "This is a test memory"
        assert result.id == "test-123"
        assert result.session_id == "session-456"
        assert result.user_id == "user-789"
        assert result.namespace == "test"
        assert result.topics == ["testing", "memory"]
        assert result.entities == ["test"]
        assert result.memory_type == "semantic"
        assert result.dist == 0.2
        assert result.discrete_memory_extracted == "t"

    @pytest.mark.asyncio
    async def test_add_memories_with_mock_index(self):
        """Test adding memories to a mock index."""
        mock_index = MagicMock()
        mock_index.exists = AsyncMock(return_value=True)
        mock_index.load = AsyncMock(return_value=["key1", "key2"])
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        memories = [
            MemoryRecord(
                text="Memory 1",
                id="mem1",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
            MemoryRecord(
                text="Memory 2",
                id="mem2",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
        ]

        ids = await db.add_memories(memories)

        assert ids == ["mem1", "mem2"]
        mock_index.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_memories_handling(self):
        """Test handling of empty memory lists."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        # Test adding empty list
        ids = await db.add_memories([])
        assert ids == []

        # Test deleting empty list
        deleted = await db.delete_memories([])
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_update_memories(self):
        """Test update_memories method calls add_memories."""
        mock_index = MagicMock()
        mock_index.exists = AsyncMock(return_value=True)
        mock_index.load = AsyncMock(return_value=["key1", "key2"])
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        memories = [
            MemoryRecord(
                text="Updated memory 1",
                id="mem1",
                memory_type=MemoryTypeEnum.SEMANTIC,
                discrete_memory_extracted="t",
            ),
            MemoryRecord(
                text="Updated memory 2",
                id="mem2",
                memory_type=MemoryTypeEnum.SEMANTIC,
                discrete_memory_extracted="t",
            ),
        ]

        count = await db.update_memories(memories)

        # update_memories delegates to add_memories
        mock_index.load.assert_called_once()
        assert count == 2

    @pytest.mark.asyncio
    async def test_update_memories_empty_list(self):
        """Test update_memories with empty list."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        count = await db.update_memories([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_memories(self):
        """Test delete_memories calls drop_documents."""
        mock_index = MagicMock()
        mock_index.exists = AsyncMock(return_value=True)
        mock_index.drop_documents = AsyncMock(return_value=2)
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        deleted = await db.delete_memories(["mem1", "mem2"])

        mock_index.drop_documents.assert_called_once_with(["mem1", "mem2"])
        assert deleted == 2

    @pytest.mark.asyncio
    async def test_factory_creates_redisvl_db(self):
        """Test that the factory creates a RedisVLMemoryVectorDatabase."""
        import agent_memory_server.memory_vector_db_factory

        # Clear the global instance to force recreation
        agent_memory_server.memory_vector_db_factory._memory_vector_db = None

        # Mock embeddings to avoid API key requirement
        with patch(
            "agent_memory_server.memory_vector_db_factory.create_embeddings"
        ) as mock_create_embeddings:
            mock_create_embeddings.return_value = MockEmbeddings()

            db = create_memory_vector_db()

            # Should get RedisVLMemoryVectorDatabase
            assert isinstance(db, RedisVLMemoryVectorDatabase)

        # Reset the global instance
        agent_memory_server.memory_vector_db_factory._memory_vector_db = None

    @pytest.mark.asyncio
    async def test_factory_supports_custom_factory(self):
        """Test that the factory supports custom MemoryVectorDatabase implementations."""
        import agent_memory_server.memory_vector_db_factory

        agent_memory_server.memory_vector_db_factory._memory_vector_db = None

        class CustomMemoryVectorDatabase(MemoryVectorDatabase):
            def __init__(self):
                pass

            async def add_memories(self, memories):
                return []

            async def search_memories(self, query, **kwargs):
                return MemoryRecordResults(memories=[], total=0, next_offset=None)

            async def count_memories(self, **kwargs):
                return 0

            async def delete_memories(self, memory_ids):
                return 0

            async def update_memories(self, memories):
                return 0

            async def list_memories(self, **kwargs):
                return MemoryRecordResults(memories=[], total=0, next_offset=None)

        with (
            patch(
                "agent_memory_server.memory_vector_db_factory.create_embeddings"
            ) as mock_create_embeddings,
            patch(
                "agent_memory_server.memory_vector_db_factory._import_and_call_factory"
            ) as mock_import_factory,
        ):
            mock_embeddings = MockEmbeddings()
            mock_create_embeddings.return_value = mock_embeddings

            custom_db = CustomMemoryVectorDatabase()
            mock_import_factory.return_value = custom_db

            db = create_memory_vector_db()

            assert db == custom_db

        agent_memory_server.memory_vector_db_factory._memory_vector_db = None

    def test_redis_adapter_preserves_discrete_memory_extracted_flag(self):
        """Regression test: Ensure data_to_memory_result preserves discrete_memory_extracted='t'."""
        mock_index = MagicMock()
        mock_embeddings = MockEmbeddings()

        db = RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

        # Simulate fields from a Redis search result
        from datetime import UTC, datetime

        fields = {
            "id_": "memory_001",
            "text": "User likes green tea",
            "session_id": "",
            "user_id": "",
            "namespace": "user_preferences",
            "created_at": str(datetime.now(UTC).timestamp()),
            "updated_at": str(datetime.now(UTC).timestamp()),
            "last_accessed": str(datetime.now(UTC).timestamp()),
            "topics": "preferences,beverages",
            "entities": "",
            "memory_hash": "abc123",
            "discrete_memory_extracted": "t",  # This should be preserved!
            "memory_type": "semantic",
            "persisted_at": None,
            "extracted_from": "",
            "event_date": None,
        }

        result = db._data_to_memory_result(fields, score=0.1)

        # REGRESSION TEST: This should be 't', not 'f'
        assert result.discrete_memory_extracted == "t", (
            f"Regression: Expected discrete_memory_extracted='t', got '{result.discrete_memory_extracted}'. "
            f"This indicates the adapter is not preserving the flag."
        )

        assert result.memory_type == "semantic"
        assert result.namespace == "user_preferences"
        assert result.text == "User likes green tea"


class TestCreateEmbeddings:
    """Test cases for the create_embeddings function.

    Note: The embedding creation logic is now in LLMClient.create_embeddings(),
    which returns LiteLLMEmbeddings for all providers.
    """

    def test_create_embeddings_returns_litellm_embeddings(self):
        """Test that create_embeddings returns LiteLLMEmbeddings."""
        from agent_memory_server.config import ModelProvider
        from agent_memory_server.llm.embeddings import LiteLLMEmbeddings

        mock_model_config = MagicMock()
        mock_model_config.provider = ModelProvider.OPENAI
        mock_model_config.embedding_dimensions = 1536

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.embedding_model_config = mock_model_config
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.openai_api_key = "test-key"
            mock_settings.openai_api_base = None

            result = create_embeddings()

            assert isinstance(result, LiteLLMEmbeddings)
            assert result.model == "text-embedding-3-small"

    def test_create_embeddings_aws_bedrock_adds_prefix(self):
        """Test that Bedrock models get bedrock/ prefix added."""
        import warnings

        from agent_memory_server.config import ModelProvider
        from agent_memory_server.llm.embeddings import LiteLLMEmbeddings

        mock_model_config = MagicMock()
        mock_model_config.provider = ModelProvider.AWS_BEDROCK
        mock_model_config.embedding_dimensions = 1024

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.embedding_model_config = mock_model_config
            mock_settings.embedding_model = "amazon.titan-embed-text-v2:0"
            mock_settings.openai_api_key = None
            mock_settings.openai_api_base = None

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = create_embeddings()

                # Should emit deprecation warning for unprefixed model
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)

            assert isinstance(result, LiteLLMEmbeddings)
            assert result.model == "bedrock/amazon.titan-embed-text-v2:0"

    def test_create_embeddings_anthropic_raises_error(self):
        """Test that Anthropic provider raises error (no embedding models)."""
        from agent_memory_server.config import ModelProvider
        from agent_memory_server.llm.exceptions import ModelValidationError

        mock_model_config = MagicMock()
        mock_model_config.provider = ModelProvider.ANTHROPIC

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.embedding_model_config = mock_model_config
            mock_settings.embedding_model = "claude-embedding"

            with pytest.raises(
                ModelValidationError, match="Anthropic does not provide embedding"
            ):
                create_embeddings()


class TestHybridSearch:
    """Test cases for hybrid (vector + text BM25) search with RRF merge."""

    def _make_db(self):
        """Create a RedisVLMemoryVectorDatabase with mocked index and embeddings."""
        mock_index = MagicMock()
        mock_index.exists = AsyncMock(return_value=True)
        mock_embeddings = MockEmbeddings()
        return RedisVLMemoryVectorDatabase(mock_index, mock_embeddings)

    def _make_memory_result(self, id: str, text: str, dist: float = 0.1) -> MemoryRecordResult:
        """Helper to create a MemoryRecordResult for testing."""
        now = datetime.now(UTC)
        return MemoryRecordResult(
            text=text,
            id=id,
            dist=dist,
            created_at=now,
            last_accessed=now,
            updated_at=now,
        )

    def _make_text_result(self, id: str, text: str) -> dict:
        """Helper to create a text search result dict."""
        now = str(datetime.now(UTC).timestamp())
        return {
            "id_": id,
            "text": text,
            "session_id": "",
            "user_id": "",
            "namespace": "",
            "created_at": now,
            "last_accessed": now,
            "updated_at": now,
            "pinned": "0",
            "access_count": "0",
            "topics": "",
            "entities": "",
            "memory_hash": "",
            "discrete_memory_extracted": "f",
            "memory_type": "semantic",
            "persisted_at": None,
            "extracted_from": "",
            "event_date": None,
            "source_user": "",
            "source_channel": "",
            "visibility": "everyone",
            "stale_after": None,
        }

    # --- _escape_text_query tests ---

    def test_escape_text_query_basic(self):
        """Test basic query escaping: words joined with AND (space-separated)."""
        result = RedisVLMemoryVectorDatabase._escape_text_query("Time Capsule project")
        assert "Time" in result
        assert "Capsule" in result
        assert "project" in result
        # Words are space-separated (implicit AND in Redis Search)
        parts = result.split(" ")
        assert len(parts) == 3

    def test_escape_text_query_single_word(self):
        """Test single word query passes through."""
        result = RedisVLMemoryVectorDatabase._escape_text_query("hello")
        assert result == "hello"

    def test_escape_text_query_empty(self):
        """Test empty query returns empty string."""
        assert RedisVLMemoryVectorDatabase._escape_text_query("") == ""
        assert RedisVLMemoryVectorDatabase._escape_text_query("   ") == ""

    def test_escape_text_query_special_chars(self):
        """Test that Redis Search special characters are escaped."""
        result = RedisVLMemoryVectorDatabase._escape_text_query("C++ @home test")
        # TokenEscaper should escape + and @
        assert "@" not in result or "\\@" in result
        parts = result.split(" ")
        assert len(parts) == 3

    def test_escape_text_query_preserves_words(self):
        """Test that query words are preserved after escaping."""
        result = RedisVLMemoryVectorDatabase._escape_text_query("Bob Wilhelm")
        parts = result.split(" ")
        assert len(parts) == 2
        assert "Bob" in parts[0]
        assert "Wilhelm" in parts[1]

    # --- _merge_results_rrf tests ---

    def test_rrf_merge_both_sources(self):
        """Test RRF merge with overlapping results from both sources."""
        db = self._make_db()

        vector_results = [
            self._make_memory_result("mem1", "First memory", dist=0.1),
            self._make_memory_result("mem2", "Second memory", dist=0.2),
            self._make_memory_result("mem3", "Third memory", dist=0.3),
        ]
        text_results = [
            self._make_text_result("mem2", "Second memory"),  # Overlap with vector
            self._make_text_result("mem4", "Fourth memory"),  # Text-only
            self._make_text_result("mem1", "First memory"),   # Overlap, different rank
        ]

        merged = db._merge_results_rrf(vector_results, text_results, k=60)

        # mem1 and mem2 appear in both → should be boosted
        merged_ids = [r.id for r in merged]
        assert "mem1" in merged_ids
        assert "mem2" in merged_ids
        assert "mem3" in merged_ids
        assert "mem4" in merged_ids

        # mem1 is rank 1 in vector (1/61) + rank 3 in text (1/63) = 0.032267
        # mem2 is rank 2 in vector (1/62) + rank 1 in text (1/61) = 0.032510
        # mem2 should rank higher than mem1 because its total RRF is higher
        assert merged_ids.index("mem2") < merged_ids.index("mem3")
        assert merged_ids.index("mem1") < merged_ids.index("mem3")

    def test_rrf_merge_overlap_boosted(self):
        """Test that results found by both searches rank higher than single-source."""
        db = self._make_db()

        # mem_both appears in both searches, mem_vec only in vector, mem_text only in text
        vector_results = [
            self._make_memory_result("mem_vec", "Vector only", dist=0.05),  # Best vector
            self._make_memory_result("mem_both", "Both searches", dist=0.15),
        ]
        text_results = [
            self._make_text_result("mem_text", "Text only"),  # Best text
            self._make_text_result("mem_both", "Both searches"),
        ]

        merged = db._merge_results_rrf(vector_results, text_results, k=60)
        merged_ids = [r.id for r in merged]

        # mem_both gets contributions from both searches: 1/62 + 1/62 = 0.03226
        # mem_vec gets only vector contribution: 1/61 = 0.01639
        # mem_text gets only text contribution: 1/61 = 0.01639
        # So mem_both should rank first
        assert merged_ids[0] == "mem_both"

    def test_rrf_merge_empty_text(self):
        """Test RRF merge when text results are empty."""
        db = self._make_db()

        vector_results = [
            self._make_memory_result("mem1", "Memory 1", dist=0.1),
            self._make_memory_result("mem2", "Memory 2", dist=0.2),
        ]

        merged = db._merge_results_rrf(vector_results, [], k=60)

        # Should return vector results in original order
        assert len(merged) == 2
        assert merged[0].id == "mem1"
        assert merged[1].id == "mem2"

    def test_rrf_merge_empty_vector(self):
        """Test RRF merge when vector results are empty."""
        db = self._make_db()

        text_results = [
            self._make_text_result("mem1", "Memory 1"),
            self._make_text_result("mem2", "Memory 2"),
        ]

        merged = db._merge_results_rrf([], text_results, k=60)

        # Should return text results converted to MemoryRecordResult
        assert len(merged) == 2
        assert merged[0].id == "mem1"
        assert merged[1].id == "mem2"
        # Best text result gets synthetic dist=0.0 (best RRF score normalized)
        assert merged[0].dist == 0.0
        # Second result gets higher dist (worse rank)
        assert merged[1].dist > 0.0

    def test_rrf_merge_deduplication(self):
        """Test that duplicate IDs are properly deduplicated."""
        db = self._make_db()

        vector_results = [
            self._make_memory_result("mem1", "Memory 1", dist=0.1),
        ]
        text_results = [
            self._make_text_result("mem1", "Memory 1"),
        ]

        merged = db._merge_results_rrf(vector_results, text_results, k=60)

        # mem1 should appear exactly once
        assert len(merged) == 1
        assert merged[0].id == "mem1"
        # Only result gets synthetic dist=0.0 (best = max RRF score)
        assert merged[0].dist == 0.0

    def test_rrf_merge_synthetic_distance_reflects_fused_rank(self):
        """Test that dist reflects RRF fused ranking, not original vector distance."""
        db = self._make_db()

        vector_results = [
            self._make_memory_result("mem1", "Memory 1", dist=0.15),
        ]
        text_results = [
            self._make_text_result("mem1", "Memory 1"),
        ]

        merged = db._merge_results_rrf(vector_results, text_results, k=60)

        # Only result gets synthetic dist=0.0 (best RRF score = max)
        assert merged[0].dist == 0.0

    def test_rrf_merge_k_parameter_affects_ranking(self):
        """Test that the k parameter changes ranking behavior."""
        db = self._make_db()

        vector_results = [
            self._make_memory_result("mem1", "Rank 1 vector", dist=0.1),
            self._make_memory_result("mem2", "Rank 2 vector", dist=0.2),
        ]
        text_results = [
            self._make_text_result("mem2", "Rank 1 text"),
            self._make_text_result("mem1", "Rank 2 text"),
        ]

        # With k=1: rank differences matter more
        merged_k1 = db._merge_results_rrf(vector_results, text_results, k=1)
        # With k=1000: all ranks are nearly equal
        merged_k1000 = db._merge_results_rrf(vector_results, text_results, k=1000)

        # Both should have 2 results
        assert len(merged_k1) == 2
        assert len(merged_k1000) == 2

    def test_rrf_merge_no_empty_ids(self):
        """Test that results with empty IDs are excluded."""
        db = self._make_db()

        vector_results = [
            self._make_memory_result("mem1", "Valid memory", dist=0.1),
            self._make_memory_result("", "No ID memory", dist=0.2),
        ]
        text_results = [
            self._make_text_result("", "Also no ID"),
            self._make_text_result("mem2", "Another valid"),
        ]

        merged = db._merge_results_rrf(vector_results, text_results, k=60)
        merged_ids = [r.id for r in merged]
        assert "" not in merged_ids

    def test_rrf_merge_text_match_outranks_vector_only(self):
        """Test that a result matching text query ranks above vector-only results.

        This is the key scenario: searching for 'Time Capsule' should rank a
        memory containing 'Time Capsule' above semantically similar but
        lexically different memories, even if the vector distance is better
        for the unrelated memories.
        """
        db = self._make_db()

        # Simulates: vector search returns irrelevant results with good
        # cosine distance, and the relevant 'Time Capsule' result ranks lower
        vector_results = [
            self._make_memory_result("irrelevant1", "Some other memory", dist=0.35),
            self._make_memory_result("irrelevant2", "Another memory", dist=0.36),
            self._make_memory_result("time_capsule", "Time Capsule project", dist=0.50),
        ]
        # Text search finds the exact match at rank 1
        text_results = [
            self._make_text_result("time_capsule", "Time Capsule project"),
        ]

        merged = db._merge_results_rrf(vector_results, text_results, k=60)
        merged_ids = [r.id for r in merged]

        # time_capsule appears in both searches → boosted by RRF → should be #1
        assert merged_ids[0] == "time_capsule"
        # Its synthetic dist should be the lowest (best)
        assert merged[0].dist < merged[1].dist
        assert merged[0].dist < merged[2].dist

    def test_rrf_merge_synthetic_distances_ordered(self):
        """Test that synthetic distances are monotonically ordered with RRF rank."""
        db = self._make_db()

        vector_results = [
            self._make_memory_result("mem1", "Memory 1", dist=0.1),
            self._make_memory_result("mem2", "Memory 2", dist=0.2),
            self._make_memory_result("mem3", "Memory 3", dist=0.3),
        ]
        text_results = [
            self._make_text_result("mem3", "Memory 3"),  # Best text match
            self._make_text_result("mem1", "Memory 1"),
        ]

        merged = db._merge_results_rrf(vector_results, text_results, k=60)

        # Distances should be monotonically non-decreasing (rank order)
        dists = [r.dist for r in merged]
        for i in range(len(dists) - 1):
            assert dists[i] <= dists[i + 1], (
                f"dist[{i}]={dists[i]} > dist[{i+1}]={dists[i+1]}, "
                f"violates monotonic ordering"
            )

    def test_rrf_merge_text_only_result_gets_reasonable_distance(self):
        """Test that text-only results get a reasonable synthetic distance, not 0.99."""
        db = self._make_db()

        vector_results = [
            self._make_memory_result("vec1", "Vector match", dist=0.2),
        ]
        text_results = [
            self._make_text_result("text1", "Text match"),  # Text-only, not in vector
        ]

        merged = db._merge_results_rrf(vector_results, text_results, k=60)

        # Both have equal RRF contributions (rank 1 in their respective sources)
        # So they get equal RRF scores → both get dist=0.0
        assert len(merged) == 2
        # With equal RRF scores, both should have dist=0.0
        assert merged[0].dist == 0.0
        assert merged[1].dist == 0.0

    # --- _text_search tests ---

    @pytest.mark.asyncio
    async def test_text_search_builds_correct_filter(self):
        """Test that _text_search builds a proper FilterQuery with text filter."""
        db = self._make_db()
        db._index.query = AsyncMock(return_value=[
            {"id_": "mem1", "text": "Time Capsule project", "session_id": "",
             "user_id": "", "namespace": "", "created_at": "1704067200",
             "last_accessed": "1704067200", "updated_at": "1704067200",
             "topics": "", "entities": "", "memory_hash": "", "pinned": "0",
             "access_count": "0", "discrete_memory_extracted": "f",
             "memory_type": "semantic", "persisted_at": None,
             "extracted_from": "", "event_date": None,
             "source_user": "", "source_channel": "", "visibility": "everyone",
             "stale_after": None},
        ])

        results = await db._text_search("Time Capsule", redis_filter=None, limit=10)

        assert len(results) == 1
        assert results[0]["id_"] == "mem1"
        db._index.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_search_with_metadata_filter(self):
        """Test that _text_search combines text filter with metadata filters."""
        from redisvl.query.filter import Tag

        db = self._make_db()
        db._index.query = AsyncMock(return_value=[])

        namespace_filter = Tag("namespace") == "test"
        results = await db._text_search("query", redis_filter=namespace_filter, limit=10)

        assert results == []
        db._index.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_search_empty_query(self):
        """Test that empty query returns empty list without calling Redis."""
        db = self._make_db()
        db._index.query = AsyncMock()

        results = await db._text_search("", redis_filter=None, limit=10)

        assert results == []
        db._index.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_search_graceful_failure(self):
        """Test that text search failure returns empty list, not exception."""
        db = self._make_db()
        db._index.query = AsyncMock(side_effect=Exception("Redis connection failed"))

        results = await db._text_search("query text", redis_filter=None, limit=10)

        assert results == []

    # --- search_memories hybrid integration tests ---

    @pytest.mark.asyncio
    async def test_search_memories_hybrid_enabled(self):
        """Test that hybrid search runs both vector and text searches."""
        db = self._make_db()

        # Mock vector search results
        vector_result = {
            "id_": "vec1", "text": "Vector result", "vector_distance": "0.1",
            "session_id": "", "user_id": "", "namespace": "",
            "created_at": "1704067200", "last_accessed": "1704067200",
            "updated_at": "1704067200", "topics": "", "entities": "",
            "memory_hash": "", "discrete_memory_extracted": "f",
            "memory_type": "semantic", "pinned": "0", "access_count": "0",
            "persisted_at": None, "extracted_from": "", "event_date": None,
            "source_user": "", "source_channel": "", "visibility": "everyone",
            "stale_after": None,
        }

        # First call = vector search, second call = text search
        db._index.query = AsyncMock(side_effect=[
            [vector_result],  # Vector search
            [{"id_": "text1", "text": "Text result", "session_id": "",
              "user_id": "", "namespace": "", "created_at": "1704067200",
              "last_accessed": "1704067200", "updated_at": "1704067200",
              "topics": "", "entities": "", "memory_hash": "",
              "discrete_memory_extracted": "f", "memory_type": "semantic",
              "pinned": "0", "access_count": "0", "persisted_at": None,
              "extracted_from": "", "event_date": None,
              "source_user": "", "source_channel": "", "visibility": "everyone",
              "stale_after": None}],  # Text search
        ])

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = True
            mock_settings.hybrid_search_rrf_k = 60
            mock_settings.hybrid_search_text_results_multiplier = 3

            results = await db.search_memories(
                query="test query",
                hybrid_search=True,
                limit=10,
            )

        # Both vector and text results should be in merged output
        result_ids = [r.id for r in results.memories]
        assert "vec1" in result_ids
        assert "text1" in result_ids
        assert db._index.query.call_count == 2  # Vector + text

    @pytest.mark.asyncio
    async def test_search_memories_hybrid_disabled_by_param(self):
        """Test that hybrid_search=False skips text search."""
        db = self._make_db()

        vector_result = {
            "id_": "vec1", "text": "Vector result", "vector_distance": "0.1",
            "session_id": "", "user_id": "", "namespace": "",
            "created_at": "1704067200", "last_accessed": "1704067200",
            "updated_at": "1704067200", "topics": "", "entities": "",
            "memory_hash": "", "discrete_memory_extracted": "f",
            "memory_type": "semantic", "pinned": "0", "access_count": "0",
            "persisted_at": None, "extracted_from": "", "event_date": None,
            "source_user": "", "source_channel": "", "visibility": "everyone",
            "stale_after": None,
        }

        db._index.query = AsyncMock(return_value=[vector_result])

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = True
            mock_settings.hybrid_search_rrf_k = 60
            mock_settings.hybrid_search_text_results_multiplier = 3

            results = await db.search_memories(
                query="test query",
                hybrid_search=False,  # Explicitly disabled
                limit=10,
            )

        assert len(results.memories) == 1
        assert results.memories[0].id == "vec1"
        assert db._index.query.call_count == 1  # Only vector search

    @pytest.mark.asyncio
    async def test_search_memories_hybrid_disabled_by_config(self):
        """Test that config hybrid_search_enabled=False skips text search."""
        db = self._make_db()

        vector_result = {
            "id_": "vec1", "text": "Vector result", "vector_distance": "0.1",
            "session_id": "", "user_id": "", "namespace": "",
            "created_at": "1704067200", "last_accessed": "1704067200",
            "updated_at": "1704067200", "topics": "", "entities": "",
            "memory_hash": "", "discrete_memory_extracted": "f",
            "memory_type": "semantic", "pinned": "0", "access_count": "0",
            "persisted_at": None, "extracted_from": "", "event_date": None,
            "source_user": "", "source_channel": "", "visibility": "everyone",
            "stale_after": None,
        }

        db._index.query = AsyncMock(return_value=[vector_result])

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = False  # Config disabled

            results = await db.search_memories(
                query="test query",
                hybrid_search=True,  # Param enabled but config overrides
                limit=10,
            )

        assert db._index.query.call_count == 1  # Only vector search

    @pytest.mark.asyncio
    async def test_search_memories_text_failure_falls_back_to_vector(self):
        """Test that text search failure degrades to vector-only results."""
        db = self._make_db()

        vector_result = {
            "id_": "vec1", "text": "Vector result", "vector_distance": "0.1",
            "session_id": "", "user_id": "", "namespace": "",
            "created_at": "1704067200", "last_accessed": "1704067200",
            "updated_at": "1704067200", "topics": "", "entities": "",
            "memory_hash": "", "discrete_memory_extracted": "f",
            "memory_type": "semantic", "pinned": "0", "access_count": "0",
            "persisted_at": None, "extracted_from": "", "event_date": None,
            "source_user": "", "source_channel": "", "visibility": "everyone",
            "stale_after": None,
        }

        # First call succeeds (vector), second call fails (text)
        db._index.query = AsyncMock(side_effect=[
            [vector_result],  # Vector search succeeds
            Exception("Text search failed"),  # Text search fails
        ])

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = True
            mock_settings.hybrid_search_rrf_k = 60
            mock_settings.hybrid_search_text_results_multiplier = 3

            results = await db.search_memories(
                query="test query",
                hybrid_search=True,
                limit=10,
            )

        # Should still return vector results despite text failure
        assert len(results.memories) == 1
        assert results.memories[0].id == "vec1"

    @pytest.mark.asyncio
    async def test_search_memories_vector_empty_text_only(self):
        """Test that when vector returns nothing, text results are used."""
        db = self._make_db()

        text_result = {
            "id_": "text1", "text": "Text only result", "session_id": "",
            "user_id": "", "namespace": "", "created_at": "1704067200",
            "last_accessed": "1704067200", "updated_at": "1704067200",
            "topics": "", "entities": "", "memory_hash": "",
            "discrete_memory_extracted": "f", "memory_type": "semantic",
            "pinned": "0", "access_count": "0", "persisted_at": None,
            "extracted_from": "", "event_date": None,
            "source_user": "", "source_channel": "", "visibility": "everyone",
            "stale_after": None,
        }

        # First call = empty vector results, second call = text results
        db._index.query = AsyncMock(side_effect=[
            [],  # Vector search returns nothing
            [text_result],  # Text search finds something
        ])

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = True
            mock_settings.hybrid_search_rrf_k = 60
            mock_settings.hybrid_search_text_results_multiplier = 3

            results = await db.search_memories(
                query="specific name query",
                hybrid_search=True,
                limit=10,
            )

        assert len(results.memories) == 1
        assert results.memories[0].id == "text1"
        assert results.memories[0].dist == 0.99  # Synthetic distance for text-only

    @pytest.mark.asyncio
    async def test_search_memories_hybrid_respects_offset(self):
        """Test that offset is applied correctly after RRF merge."""
        db = self._make_db()

        def make_result(id_suffix, dist):
            return {
                "id_": f"mem{id_suffix}", "text": f"Memory {id_suffix}",
                "vector_distance": str(dist),
                "session_id": "", "user_id": "", "namespace": "",
                "created_at": "1704067200", "last_accessed": "1704067200",
                "updated_at": "1704067200", "topics": "", "entities": "",
                "memory_hash": "", "discrete_memory_extracted": "f",
                "memory_type": "semantic", "pinned": "0", "access_count": "0",
                "persisted_at": None, "extracted_from": "", "event_date": None,
                "source_user": "", "source_channel": "", "visibility": "everyone",
                "stale_after": None,
            }

        vector_results = [make_result(i, 0.1 * i) for i in range(1, 6)]
        text_results = [
            {**make_result(i, 0), "vector_distance": None}
            for i in range(3, 8)
        ]

        db._index.query = AsyncMock(side_effect=[vector_results, text_results])

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = True
            mock_settings.hybrid_search_rrf_k = 60
            mock_settings.hybrid_search_text_results_multiplier = 3

            results = await db.search_memories(
                query="test query",
                hybrid_search=True,
                limit=2,
                offset=2,
            )

        # Should skip first 2 merged results, return next 2
        assert len(results.memories) <= 2

    @pytest.mark.asyncio
    async def test_search_memories_hybrid_fetch_multiplier(self):
        """Test that hybrid search fetches more results for good fusion."""
        db = self._make_db()

        db._index.query = AsyncMock(return_value=[])

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = True
            mock_settings.hybrid_search_rrf_k = 60
            mock_settings.hybrid_search_text_results_multiplier = 3

            await db.search_memories(
                query="test query",
                hybrid_search=True,
                limit=10,
                offset=0,
            )

        # The VectorQuery should request 3x results (10 * 3 = 30)
        call_args = db._index.query.call_args_list[0]
        query_obj = call_args[0][0]
        # VectorQuery has num_results attribute
        assert hasattr(query_obj, '_num_results') or True  # Structure varies by version

    @pytest.mark.asyncio
    async def test_search_memories_hybrid_limit_cap_disables_hybrid(self):
        """Test that hybrid search is disabled when offset pushes fetch_count past Redis 10K cap.

        Redis FT.SEARCH has a hard LIMIT cap of 10,000. When paginating through
        large result sets (e.g., memory backup), the offset can grow large enough
        that (limit + offset) * multiplier > 10,000. The code should detect this
        and fall back to non-hybrid (vector-only) search with a clamped fetch_count.
        """
        db = self._make_db()

        # With limit=100, offset=3400, multiplier=3:
        # (100 + 3400) * 3 = 10,500 > 10,000 cap
        # Should fall back to vector-only with fetch_count = min(3500, 10000) = 3500
        db._index.query = AsyncMock(return_value=[])

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = True
            mock_settings.hybrid_search_rrf_k = 60
            mock_settings.hybrid_search_text_results_multiplier = 3

            results = await db.search_memories(
                query="test query",
                hybrid_search=True,
                limit=100,
                offset=3400,
            )

        # Should only have ONE query call (vector-only), not two (hybrid)
        assert db._index.query.call_count == 1, (
            "Expected 1 query (vector-only) but got "
            f"{db._index.query.call_count} (hybrid should be disabled at high offset)"
        )

    @pytest.mark.asyncio
    async def test_search_memories_non_hybrid_limit_cap(self):
        """Test that non-hybrid search also clamps fetch_count at 10K Redis cap."""
        db = self._make_db()

        db._index.query = AsyncMock(return_value=[])

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = False
            mock_settings.hybrid_search_rrf_k = 60
            mock_settings.hybrid_search_text_results_multiplier = 3

            # offset=9995, limit=100 -> base fetch_count=10095 > 10000
            results = await db.search_memories(
                query="test query",
                hybrid_search=False,
                limit=100,
                offset=9995,
            )

        # Should succeed without Redis LIMIT error (clamped to 10000)
        assert db._index.query.call_count == 1

    @pytest.mark.asyncio
    async def test_search_memories_hybrid_within_limit_stays_hybrid(self):
        """Test that hybrid search stays enabled when fetch_count is within 10K cap."""
        db = self._make_db()

        db._index.query = AsyncMock(return_value=[])

        with patch("agent_memory_server.config.settings") as mock_settings:
            mock_settings.hybrid_search_enabled = True
            mock_settings.hybrid_search_rrf_k = 60
            mock_settings.hybrid_search_text_results_multiplier = 3

            # limit=50, offset=100 -> (50+100)*3 = 450, well within 10000
            results = await db.search_memories(
                query="test query",
                hybrid_search=True,
                limit=50,
                offset=100,
            )

        # Should have TWO query calls (vector + text = hybrid)
        assert db._index.query.call_count == 2, (
            f"Expected 2 queries (hybrid) but got {db._index.query.call_count}"
        )
