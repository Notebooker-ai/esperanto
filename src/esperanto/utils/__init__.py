"""Utility modules for Esperanto."""

from esperanto.utils.embedding import validate_and_decode_embedding
from esperanto.utils.model_cache import ModelCache
from esperanto.utils.token_utils import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_OUTPUT_TOKENS,
    OUTPUT_RATIO,
    SAFETY_BUFFER,
    batch_by_token_limit,
    calculate_batch_token_limit,
    calculate_output_buffer,
    chunk_text_by_tokens,
    get_context_limit_from_error,
    is_context_limit_error,
    parse_context_limit_error,
    token_count,
)

__all__ = [
    "ModelCache",
    "validate_and_decode_embedding",
    "DEFAULT_CONTEXT_LIMIT",
    "DEFAULT_OUTPUT_TOKENS",
    "OUTPUT_RATIO",
    "SAFETY_BUFFER",
    "batch_by_token_limit",
    "calculate_batch_token_limit",
    "calculate_output_buffer",
    "chunk_text_by_tokens",
    "get_context_limit_from_error",
    "is_context_limit_error",
    "parse_context_limit_error",
    "token_count",
]
