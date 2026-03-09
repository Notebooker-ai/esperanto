"""Tests for adaptive retry in embedding base class."""

import os
import pytest
from dataclasses import dataclass
from typing import List
from unittest.mock import patch

from esperanto.common_types import Model
from esperanto.providers.embedding.base import EmbeddingModel


@dataclass
class MockEmbeddingModel(EmbeddingModel):
    """Concrete embedding model for testing."""

    def __post_init__(self):
        super().__post_init__()

    @property
    def provider(self) -> str:
        return "mock"

    def _get_models(self) -> List[Model]:
        return []

    def _get_default_model(self) -> str:
        return "mock-model"

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]

    async def aembed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]


class TestDefaultConfig:
    def test_retry_on_by_default(self):
        model = MockEmbeddingModel()
        assert model.auto_retry_on_context_limit is True

    def test_retry_disabled_via_config(self):
        model = MockEmbeddingModel(
            config={"auto_retry_on_context_limit": False}
        )
        assert model.auto_retry_on_context_limit is False

    def test_retry_enabled_via_config(self):
        model = MockEmbeddingModel(
            config={"auto_retry_on_context_limit": True}
        )
        assert model.auto_retry_on_context_limit is True

    def test_max_retry_depth_default(self):
        model = MockEmbeddingModel()
        assert model.max_retry_depth == 3

    def test_max_retry_depth_via_config(self):
        model = MockEmbeddingModel(config={"max_retry_depth": 5})
        assert model.max_retry_depth == 5


class TestEnvVarConfig:
    def test_retry_enabled_via_env(self):
        with patch.dict(os.environ, {"ESPERANTO_EMBEDDING_AUTO_RETRY": "true"}):
            model = MockEmbeddingModel()
            assert model.auto_retry_on_context_limit is True

    def test_retry_disabled_via_env(self):
        with patch.dict(os.environ, {"ESPERANTO_EMBEDDING_AUTO_RETRY": "false"}):
            model = MockEmbeddingModel()
            assert model.auto_retry_on_context_limit is False

    def test_env_overrides_default_true(self):
        """When env is set to 'false', it overrides the default of True."""
        with patch.dict(os.environ, {"ESPERANTO_EMBEDDING_AUTO_RETRY": "0"}):
            model = MockEmbeddingModel()
            assert model.auto_retry_on_context_limit is False

    def test_config_overrides_env(self):
        with patch.dict(os.environ, {"ESPERANTO_EMBEDDING_AUTO_RETRY": "true"}):
            model = MockEmbeddingModel(
                config={"auto_retry_on_context_limit": False}
            )
            assert model.auto_retry_on_context_limit is False


class TestRetryDisabled:
    def test_error_propagates_when_retry_off(self):
        """When retry is off, context limit errors propagate unchanged."""

        @dataclass
        class FailingModel(MockEmbeddingModel):
            def embed(self, texts, **kwargs):
                raise RuntimeError(
                    "This model's maximum context length is 8192 tokens. "
                    "However, your messages resulted in 10000 tokens."
                )

        model = FailingModel(config={"auto_retry_on_context_limit": False})
        with pytest.raises(RuntimeError, match="maximum context length"):
            model.embed(["test text"])

    @pytest.mark.asyncio
    async def test_async_error_propagates_when_retry_off(self):
        """When retry is off, async context limit errors propagate unchanged."""

        @dataclass
        class FailingModel(MockEmbeddingModel):
            async def aembed(self, texts, **kwargs):
                raise RuntimeError(
                    "This model's maximum context length is 8192 tokens. "
                    "However, your messages resulted in 10000 tokens."
                )

        model = FailingModel(config={"auto_retry_on_context_limit": False})
        with pytest.raises(RuntimeError, match="maximum context length"):
            await model.aembed(["test text"])


class TestRetryEnabled:
    def test_retries_with_smaller_batches(self):
        """When retry is on and context error occurs, retries with smaller batches."""
        call_count = 0

        @dataclass
        class RetryModel(MockEmbeddingModel):
            def embed(self, texts, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1 and len(texts) > 1:
                    raise RuntimeError(
                        "This model's maximum context length is 100 tokens. "
                        "However, your messages resulted in 200 tokens."
                    )
                return [[1.0, 2.0, 3.0] for _ in texts]

        model = RetryModel(config={"auto_retry_on_context_limit": True})
        result = model.embed(["text one", "text two", "text three"])
        assert len(result) == 3
        assert call_count > 1

    @pytest.mark.asyncio
    async def test_async_retries_with_smaller_batches(self):
        """Async version retries with smaller batches."""
        call_count = 0

        @dataclass
        class RetryModel(MockEmbeddingModel):
            async def aembed(self, texts, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1 and len(texts) > 1:
                    raise RuntimeError(
                        "This model's maximum context length is 100 tokens. "
                        "However, your messages resulted in 200 tokens."
                    )
                return [[1.0, 2.0, 3.0] for _ in texts]

        model = RetryModel(config={"auto_retry_on_context_limit": True})
        result = await model.aembed(["text one", "text two", "text three"])
        assert len(result) == 3
        assert call_count > 1

    def test_non_context_errors_propagate(self):
        """Non-context errors propagate even with retry enabled."""

        @dataclass
        class RateLimitModel(MockEmbeddingModel):
            def embed(self, texts, **kwargs):
                raise RuntimeError("Rate limit exceeded")

        model = RateLimitModel(config={"auto_retry_on_context_limit": True})
        with pytest.raises(RuntimeError, match="Rate limit"):
            model.embed(["test text"])

    @pytest.mark.asyncio
    async def test_async_non_context_errors_propagate(self):
        """Async non-context errors propagate even with retry enabled."""

        @dataclass
        class RateLimitModel(MockEmbeddingModel):
            async def aembed(self, texts, **kwargs):
                raise RuntimeError("Rate limit exceeded")

        model = RateLimitModel(config={"auto_retry_on_context_limit": True})
        with pytest.raises(RuntimeError, match="Rate limit"):
            await model.aembed(["test text"])


class TestMaxDepth:
    def test_max_depth_raises_runtime_error(self):
        """Exceeding max retry depth raises RuntimeError."""

        @dataclass
        class AlwaysFailModel(MockEmbeddingModel):
            def embed(self, texts, **kwargs):
                raise RuntimeError(
                    "This model's maximum context length is 100 tokens. "
                    "However, your messages resulted in 200 tokens."
                )

        model = AlwaysFailModel(
            config={"auto_retry_on_context_limit": True, "max_retry_depth": 1}
        )
        with pytest.raises(RuntimeError, match="max depth"):
            model.embed(["text one", "text two"])

    @pytest.mark.asyncio
    async def test_async_max_depth_raises_runtime_error(self):
        """Async version raises RuntimeError when max depth exceeded."""

        @dataclass
        class AlwaysFailModel(MockEmbeddingModel):
            async def aembed(self, texts, **kwargs):
                raise RuntimeError(
                    "This model's maximum context length is 100 tokens. "
                    "However, your messages resulted in 200 tokens."
                )

        model = AlwaysFailModel(
            config={"auto_retry_on_context_limit": True, "max_retry_depth": 1}
        )
        with pytest.raises(RuntimeError, match="max depth"):
            await model.aembed(["text one", "text two"])


class TestSingleTextChunking:
    def test_single_oversized_text_gets_chunked(self):
        """A single text that exceeds context gets chunked and averaged."""
        call_count = 0

        @dataclass
        class ChunkModel(MockEmbeddingModel):
            def embed(self, texts, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError(
                        "This model's maximum context length is 50 tokens. "
                        "However, your messages resulted in 200 tokens."
                    )
                return [[float(call_count)] * 3 for _ in texts]

        model = ChunkModel(config={"auto_retry_on_context_limit": True})
        long_text = "word " * 500
        result = model.embed([long_text])
        assert len(result) == 1
        assert len(result[0]) == 3

    @pytest.mark.asyncio
    async def test_async_single_oversized_text_gets_chunked(self):
        """Async: single oversized text gets chunked and averaged."""
        call_count = 0

        @dataclass
        class ChunkModel(MockEmbeddingModel):
            async def aembed(self, texts, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError(
                        "This model's maximum context length is 50 tokens. "
                        "However, your messages resulted in 200 tokens."
                    )
                return [[float(call_count)] * 3 for _ in texts]

        model = ChunkModel(config={"auto_retry_on_context_limit": True})
        long_text = "word " * 500
        result = await model.aembed([long_text])
        assert len(result) == 1
        assert len(result[0]) == 3


class TestWrapperTransparency:
    def test_embed_works_normally(self):
        """embed() works transparently when retry is off."""
        model = MockEmbeddingModel()
        result = model.embed(["hello", "world"])
        assert result == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

    @pytest.mark.asyncio
    async def test_aembed_works_normally(self):
        """aembed() works transparently when retry is off."""
        model = MockEmbeddingModel()
        result = await model.aembed(["hello", "world"])
        assert result == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]


class TestAverageEmbeddings:
    def test_average_embeddings(self):
        result = EmbeddingModel._average_embeddings(
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
        )
        assert result == [2.0, 3.0, 4.0]

    def test_average_single_embedding(self):
        result = EmbeddingModel._average_embeddings([[1.0, 2.0, 3.0]])
        assert result == [1.0, 2.0, 3.0]

    def test_average_empty(self):
        result = EmbeddingModel._average_embeddings([])
        assert result == []
