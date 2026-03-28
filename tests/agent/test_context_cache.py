"""Tests for context_cache parameter in get_model_context_length().

This tests the new context_cache feature that allows users to disable
persistent context length caching for custom providers.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.model_metadata import (
    get_model_context_length,
    save_context_length,
    get_cached_context_length,
    CONTEXT_PROBE_TIERS,
)


class TestContextCacheParameter:
    """Test the context_cache parameter behavior."""

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_context_cache_true_uses_cache(self, mock_fetch, tmp_path):
        """When context_cache=True (default), should use persistent cache."""
        mock_fetch.return_value = {"test/model": {"context_length": 999999}}
        cache_file = tmp_path / "cache.yaml"
        
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            # Save to cache
            save_context_length("test/model", "http://local:8080", 32768)
            
            # With context_cache=True (default), should return cached value
            result = get_model_context_length(
                "test/model",
                base_url="http://local:8080",
                context_cache=True,
            )
            assert result == 32768  # cached value, not API's 999999

    @patch("agent.model_metadata.fetch_endpoint_model_metadata")
    @patch("agent.model_metadata.fetch_model_metadata")
    def test_context_cache_false_skips_cache(self, mock_fetch, mock_endpoint, tmp_path):
        """When context_cache=False, should skip persistent cache."""
        mock_fetch.return_value = {}
        mock_endpoint.return_value = {"test/model": {"context_length": 65536}}
        cache_file = tmp_path / "cache.yaml"
        
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            # Save to cache
            save_context_length("test/model", "http://local:8080", 32768)
            
            # With context_cache=False, should skip cache and use endpoint metadata
            result = get_model_context_length(
                "test/model",
                base_url="http://local:8080",
                context_cache=False,
            )
            assert result == 65536  # endpoint value, not cached 32768

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_context_cache_false_with_no_api_falls_back(self, mock_fetch, tmp_path):
        """When context_cache=False and API has no data, should fall back to defaults."""
        mock_fetch.return_value = {}  # No model data
        cache_file = tmp_path / "cache.yaml"
        
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            # Save to cache
            save_context_length("unknown/model", "http://local:8080", 32768)
            
            # With context_cache=False and no API data, should use probe tier
            result = get_model_context_length(
                "unknown/model",
                base_url="http://local:8080",
                context_cache=False,
            )
            assert result == CONTEXT_PROBE_TIERS[0]  # 128000

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_context_cache_default_is_true(self, mock_fetch, tmp_path):
        """Default behavior (no context_cache param) should use cache."""
        mock_fetch.return_value = {"test/model": {"context_length": 999999}}
        cache_file = tmp_path / "cache.yaml"
        
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            # Save to cache
            save_context_length("test/model", "http://local:8080", 32768)
            
            # Without context_cache param, should default to True (use cache)
            result = get_model_context_length(
                "test/model",
                base_url="http://local:8080",
            )
            assert result == 32768  # cached value (backward compatible)

    @patch("agent.model_metadata.fetch_model_metadata")
    def test_context_cache_false_with_config_override_ignored(self, mock_fetch, tmp_path):
        """config_context_length should override context_cache setting."""
        mock_fetch.return_value = {"test/model": {"context_length": 65536}}
        cache_file = tmp_path / "cache.yaml"
        
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            # Save to cache
            save_context_length("test/model", "http://local:8080", 32768)
            
            # config_context_length should win regardless of context_cache
            result = get_model_context_length(
                "test/model",
                base_url="http://local:8080",
                config_context_length=131072,
                context_cache=False,
            )
            assert result == 131072  # config override wins


class TestContextCacheWithLocalEndpoint:
    """Test context_cache with local endpoint detection."""

    @patch("agent.model_metadata.fetch_model_metadata")
    @patch("agent.model_metadata._query_local_context_length")
    def test_context_cache_false_queries_local(self, mock_local, mock_fetch, tmp_path):
        """With context_cache=False, should query local server."""
        mock_fetch.return_value = {}
        mock_local.return_value = 131072  # Local server reports this
        cache_file = tmp_path / "cache.yaml"
        
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            # Save old value to cache
            save_context_length("qwen3.5:27b", "http://localhost:8080", 32768)
            
            # With context_cache=False, should query local and get new value
            result = get_model_context_length(
                "qwen3.5:27b",
                base_url="http://localhost:8080",
                context_cache=False,
            )
            assert result == 131072  # Fresh local query, not cached 32768
            mock_local.assert_called_once()

    @patch("agent.model_metadata.fetch_model_metadata")
    @patch("agent.model_metadata._query_local_context_length")
    def test_context_cache_true_skips_local_query(self, mock_local, mock_fetch, tmp_path):
        """With context_cache=True, should use cache and not query local."""
        mock_fetch.return_value = {}
        mock_local.return_value = 131072
        cache_file = tmp_path / "cache.yaml"
        
        with patch("agent.model_metadata._get_context_cache_path", return_value=cache_file):
            # Save to cache
            save_context_length("qwen3.5:27b", "http://localhost:8080", 32768)
            
            # With context_cache=True, should use cache and NOT query local
            result = get_model_context_length(
                "qwen3.5:27b",
                base_url="http://localhost:8080",
                context_cache=True,
            )
            assert result == 32768  # Cached value
            mock_local.assert_not_called()  # Should not have queried local server
