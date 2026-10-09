from unittest.mock import MagicMock, patch


class TestAWSUtilities:
    """Test cases for AWS utility functions."""

    def test_bedrock_embedding_model_exists_found(self):
        """Test bedrock_embedding_model_exists returns True when model is found."""
        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the cache for this test
        bedrock_embedding_model_exists.cache_clear()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "amazon.titan-embed-text-v2:0",
                    "outputModalities": ["EMBEDDING"],
                },
                {
                    "modelId": "amazon.titan-embed-text-v1",
                    "outputModalities": ["EMBEDDING"],
                },
            ]
        }

        with patch(
            "agent_memory_server._aws.utils.create_bedrock_client",
            return_value=mock_client,
        ):
            result = bedrock_embedding_model_exists(
                "amazon.titan-embed-text-v2:0",
                region_name="us-east-1",
            )

        assert result is True
        mock_client.list_foundation_models.assert_called_once_with(
            byOutputModality="EMBEDDING"
        )

    def test_bedrock_embedding_model_exists_not_found(self):
        """Test bedrock_embedding_model_exists returns False when model is not found."""
        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the cache for this test
        bedrock_embedding_model_exists.cache_clear()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "amazon.titan-embed-text-v1",
                    "outputModalities": ["EMBEDDING"],
                },
            ]
        }

        with patch(
            "agent_memory_server._aws.utils.create_bedrock_client",
            return_value=mock_client,
        ):
            result = bedrock_embedding_model_exists(
                "non-existent-model",
                region_name="us-east-1",
            )

        assert result is False

    def test_bedrock_embedding_model_exists_via_model_modality(self):
        """Test model detection via modelModality field."""
        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the cache for this test
        bedrock_embedding_model_exists.cache_clear()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "cohere.embed-english-v3",
                    "modelModality": "EMBEDDING",
                    "outputModalities": [],  # Empty but modelModality is EMBEDDING
                },
            ]
        }

        with patch(
            "agent_memory_server._aws.utils.create_bedrock_client",
            return_value=mock_client,
        ):
            result = bedrock_embedding_model_exists(
                "cohere.embed-english-v3",
                region_name="us-east-1",
            )

        assert result is True

    def test_bedrock_embedding_model_exists_client_error(self):
        """A ClientError (e.g. AccessDenied) is logged and treated as "exists".

        Deliberate since the fork's d38bd4b: this is a pre-check, not a gate, so
        misconfigured credentials surface from the real embedding call instead of
        as a misleading "model not found". (This test still asserted the old
        False until 2026-10-09.)
        """
        from botocore.exceptions import ClientError

        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the cache for this test
        bedrock_embedding_model_exists.cache_clear()

        mock_client = MagicMock()
        mock_client.list_foundation_models.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "ListFoundationModels",
        )

        with (
            patch(
                "agent_memory_server._aws.utils.create_bedrock_client",
                return_value=mock_client,
            ),
            patch("agent_memory_server._aws.utils.logger") as mock_logger,
        ):
            result = bedrock_embedding_model_exists(
                "amazon.titan-embed-text-v2:0",
                region_name="us-east-1",
            )

        assert result is True
        # The swallowed ClientError must be logged, not silent.
        mock_logger.exception.assert_called_once()
        assert "Defaulting to True" in mock_logger.exception.call_args.args[0]

    def test_bedrock_embedding_model_exists_caching(self):
        """Test that results are cached."""
        from agent_memory_server._aws.utils import bedrock_embedding_model_exists

        # Clear the cache for this test
        bedrock_embedding_model_exists.cache_clear()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "amazon.titan-embed-text-v2:0",
                    "outputModalities": ["EMBEDDING"],
                },
            ]
        }

        with patch(
            "agent_memory_server._aws.utils.create_bedrock_client",
            return_value=mock_client,
        ) as mock_create_client:
            # First call
            result1 = bedrock_embedding_model_exists(
                "amazon.titan-embed-text-v2:0",
                region_name="us-east-1",
            )
            # Second call (should be cached)
            result2 = bedrock_embedding_model_exists(
                "amazon.titan-embed-text-v2:0",
                region_name="us-east-1",
            )

        assert result1 is True
        assert result2 is True
        # Should only create client once due to caching
        assert mock_create_client.call_count == 1
