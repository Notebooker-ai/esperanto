"""Tests for esperanto.utils.token_utils."""

import pytest
from unittest.mock import patch

from esperanto.utils.token_utils import (
    SAFETY_BUFFER,
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_OUTPUT_TOKENS,
    token_count,
    parse_context_limit_error,
    is_context_limit_error,
    get_context_limit_from_error,
    batch_by_token_limit,
    calculate_batch_token_limit,
    calculate_output_buffer,
    chunk_text_by_tokens,
)


class TestTokenCount:
    def test_empty_string(self):
        assert token_count("") == 0

    def test_nonempty_string(self):
        result = token_count("Hello world")
        assert result > 0

    def test_fallback_without_tiktoken(self):
        with patch.dict("sys.modules", {"tiktoken": None}):
            # Force reimport to trigger ImportError path
            import importlib
            import esperanto.utils.token_utils as mod

            importlib.reload(mod)
            result = mod.token_count("Hello world this is a test")
            # Fallback: len(words) * 1.3 = 6 * 1.3 = 7.8 -> 7
            assert result == int(6 * 1.3)
            # Restore
            importlib.reload(mod)

    def test_with_tiktoken_available(self):
        try:
            import tiktoken  # noqa: F401

            result = token_count("Hello world")
            assert isinstance(result, int)
            assert result > 0
        except ImportError:
            pytest.skip("tiktoken not installed")


class TestParseContextLimitError:
    def test_openai_format(self):
        error = RuntimeError(
            "This model's maximum context length is 8192 tokens. "
            "However, your messages resulted in 10000 tokens."
        )
        result = parse_context_limit_error(error)
        assert result == (10000, 8192)

    def test_openai_embedding_format(self):
        error = RuntimeError(
            "This model's maximum context length is 8192 tokens, "
            "however you requested 10000 tokens"
        )
        result = parse_context_limit_error(error)
        assert result == (10000, 8192)

    def test_anthropic_format(self):
        error = RuntimeError("prompt is too long: 10000 tokens > 8192 maximum")
        result = parse_context_limit_error(error)
        assert result == (10000, 8192)

    def test_google_format(self):
        error = RuntimeError(
            "The input token count (10000) exceeds the maximum (8192)"
        )
        result = parse_context_limit_error(error)
        assert result == (10000, 8192)

    def test_unrecognized_format(self):
        error = RuntimeError("Something went wrong")
        result = parse_context_limit_error(error)
        assert result is None

    def test_empty_message(self):
        assert parse_context_limit_error(RuntimeError("")) is None
        assert parse_context_limit_error(None) is None

    def test_generic_format(self):
        error = RuntimeError("Error: 10000 tokens > 8192")
        result = parse_context_limit_error(error)
        assert result == (10000, 8192)


class TestIsContextLimitError:
    def test_context_limit_error(self):
        error = RuntimeError(
            "This model's maximum context length is 8192 tokens"
        )
        assert is_context_limit_error(error) is True

    def test_token_limit_error(self):
        error = RuntimeError("token limit exceeded")
        assert is_context_limit_error(error) is True

    def test_prompt_too_long(self):
        error = RuntimeError("prompt is too long: 10000 tokens > 8192 maximum")
        assert is_context_limit_error(error) is True

    def test_rate_limit_not_context(self):
        error = RuntimeError("Rate limit exceeded. Please retry after 60s")
        assert is_context_limit_error(error) is False

    def test_auth_error_not_context(self):
        error = RuntimeError("Unauthorized: Invalid API key")
        assert is_context_limit_error(error) is False

    def test_timeout_not_context(self):
        error = RuntimeError("Request timeout after 30 seconds")
        assert is_context_limit_error(error) is False

    def test_generic_error_not_context(self):
        error = RuntimeError("Something went wrong")
        assert is_context_limit_error(error) is False


class TestGetContextLimitFromError:
    def test_parseable_context_error(self):
        error = RuntimeError(
            "This model's maximum context length is 8192 tokens. "
            "However, your messages resulted in 10000 tokens."
        )
        tokens_sent, context_limit = get_context_limit_from_error(error)
        assert tokens_sent == 10000
        assert context_limit == 8192

    def test_unparseable_error_uses_default(self):
        error = RuntimeError("Something went wrong")
        tokens_sent, context_limit = get_context_limit_from_error(error)
        assert tokens_sent is None
        assert context_limit == DEFAULT_CONTEXT_LIMIT

    def test_custom_default_limit(self):
        error = RuntimeError("Something went wrong")
        tokens_sent, context_limit = get_context_limit_from_error(error, 4096)
        assert tokens_sent is None
        assert context_limit == 4096


class TestBatchByTokenLimit:
    def test_empty_list(self):
        assert list(batch_by_token_limit([], 1000)) == []

    def test_all_fit_in_one_batch(self):
        texts = ["hello", "world"]
        batches = list(batch_by_token_limit(texts, 10000))
        assert len(batches) == 1
        assert batches[0] == texts

    def test_multiple_batches(self):
        # Create texts that force multiple batches
        texts = ["word " * 100 for _ in range(5)]
        batches = list(batch_by_token_limit(texts, 200))
        assert len(batches) > 1
        # All texts should be present
        all_texts = [t for batch in batches for t in batch]
        assert len(all_texts) == 5

    def test_oversized_single_text(self):
        texts = ["word " * 1000, "small"]
        batches = list(batch_by_token_limit(texts, 100))
        # Oversized text goes in its own batch
        assert any(len(b) == 1 and "word" in b[0] for b in batches)


class TestCalculateBatchTokenLimit:
    def test_within_limit(self):
        limit = calculate_batch_token_limit(5000, 8192, 3)
        assert limit == int(8192 * SAFETY_BUFFER)

    def test_exceeds_limit_distributes(self):
        limit = calculate_batch_token_limit(20000, 8192, 5)
        # Should distribute across multiple batches
        assert limit < int(8192 * SAFETY_BUFFER)


class TestCalculateOutputBuffer:
    def test_normal_limit(self):
        buffer = calculate_output_buffer(8192)
        assert buffer == int(8192 * 0.10)

    def test_small_limit(self):
        buffer = calculate_output_buffer(1000)
        assert buffer == 100

    def test_large_limit(self):
        buffer = calculate_output_buffer(100000)
        assert buffer == 10000


class TestChunkTextByTokens:
    def test_empty_text(self):
        assert chunk_text_by_tokens("", 100) == []

    def test_text_fits(self):
        text = "Short text."
        chunks = chunk_text_by_tokens(text, 1000)
        assert chunks == [text]

    def test_paragraph_splitting(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text_by_tokens(text, 5)
        assert len(chunks) > 1
        # All content should be present
        combined = " ".join(chunks)
        assert "First" in combined
        assert "Third" in combined

    def test_sentence_splitting(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text_by_tokens(text, 5)
        assert len(chunks) > 1

    def test_word_splitting(self):
        # A single long sentence with no paragraph or sentence boundaries
        text = " ".join(["word"] * 100)
        chunks = chunk_text_by_tokens(text, 10)
        assert len(chunks) > 1
        # All words should be present
        total_words = sum(c.count("word") for c in chunks)
        assert total_words == 100


class TestConstants:
    def test_safety_buffer(self):
        assert SAFETY_BUFFER == 0.90

    def test_default_context_limit(self):
        assert DEFAULT_CONTEXT_LIMIT == 8192

    def test_default_output_tokens(self):
        assert DEFAULT_OUTPUT_TOKENS == 4096
