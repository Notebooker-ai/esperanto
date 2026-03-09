"""Token utilities for Esperanto.

Provides token counting, context limit error parsing, text chunking,
and batch management for working within model context windows.
"""

import re
from typing import Iterator, List, Optional, Tuple

# Safety buffer: use 90% of context limit to avoid edge cases
SAFETY_BUFFER = 0.90

# Conservative fallback context limit (tokens)
DEFAULT_CONTEXT_LIMIT = 8192

# Initial output buffer (tokens)
DEFAULT_OUTPUT_TOKENS = 4096


def token_count(text: str) -> int:
    """Estimate the number of tokens in text.

    Uses tiktoken (o200k_base encoding) if available, otherwise falls back
    to a word-count heuristic (words * 1.3).

    Args:
        text: The text to count tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except ImportError:
        return int(len(text.split()) * 1.3)


def parse_context_limit_error(error: Exception) -> Optional[Tuple[int, int]]:
    """Extract token counts from context limit error messages.

    Supports error formats from OpenAI, Anthropic, and Google.

    Args:
        error: The exception to parse.

    Returns:
        Tuple of (tokens_sent, context_limit) if parseable, None otherwise.
    """
    error_str = str(error) if error else ""
    if not error_str:
        return None

    # Pattern: "tokens (X) exceeded...limit (Y)" or similar
    match = re.search(
        r"tokens?\s*\(?(\d+)\)?\s*(?:exceeded|>).*?(?:limit|max|maximum)[^\d]*(\d+)",
        error_str,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1)), int(match.group(2))

    # Pattern: "X tokens > Y max" format
    match = re.search(
        r"(\d+)\s*tokens?\s*(?:>|exceeded)\s*(\d+)\s*(?:max|limit)",
        error_str,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1)), int(match.group(2))

    # OpenAI format: "maximum context length is 8192 tokens...10000 tokens"
    match = re.search(
        r"maximum context length is (\d+) tokens.*?(\d+) tokens",
        error_str,
        re.DOTALL,
    )
    if match:
        return int(match.group(2)), int(match.group(1))

    # Anthropic format: "prompt is too long: 10000 tokens > 8192 maximum"
    match = re.search(
        r"prompt is too long: (\d+) tokens? > (\d+) maximum",
        error_str,
    )
    if match:
        return int(match.group(1)), int(match.group(2))

    # Google format: "input token count (10000) exceeds the maximum (8192)"
    match = re.search(
        r"input token count \((\d+)\) exceeds the maximum \((\d+)\)",
        error_str,
    )
    if match:
        return int(match.group(1)), int(match.group(2))

    # Generic format: "10000 tokens > 8192"
    match = re.search(r"(\d+)\s*tokens?\s*>\s*(\d+)", error_str)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Pattern: "maximum context length is X tokens" (only limit known)
    match = re.search(
        r"(?:max|maximum)\s*(?:context)?\s*(?:length|limit|window)?\s*(?:is|of|:)?\s*(\d+)",
        error_str,
        re.IGNORECASE,
    )
    if match:
        return None, int(match.group(1))

    # Pattern: large number near context/limit/max/window
    match = re.search(
        r"(?:context|limit|max|window)[^\d]*(\d{4,})",
        error_str,
        re.IGNORECASE,
    )
    if match:
        return None, int(match.group(1))

    return None


def is_context_limit_error(error: Exception) -> bool:
    """Check if an exception is a context/token limit error.

    Distinguishes context limit errors from rate limits, auth errors, etc.

    Args:
        error: The exception to check.

    Returns:
        True if this is a context/token limit error.
    """
    error_msg = str(error).lower()

    # Positive indicators: context limit errors
    context_keywords = [
        "maximum context length",
        "token limit",
        "too many tokens",
        "prompt is too long",
        "exceeds the maximum",
        "context window",
        "max_tokens",
        "token count",
        "context length",
        "input too long",
        "request too large",
        "payload size exceeds",
        "you requested",
    ]

    # Negative indicators: not context errors
    non_context_keywords = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "quota exceeded",
        "unauthorized",
        "authentication",
        "forbidden",
        "not found",
        "invalid api key",
        "billing",
        "insufficient_quota",
        "server error",
        "internal error",
        "timeout",
        "connection",
    ]

    # Check negative indicators first
    for keyword in non_context_keywords:
        if keyword in error_msg:
            return False

    # Check positive indicators
    for keyword in context_keywords:
        if keyword in error_msg:
            return True

    return False


def get_context_limit_from_error(
    error: Exception,
    default_limit: int = DEFAULT_CONTEXT_LIMIT,
) -> Tuple[Optional[int], int]:
    """Parse a context-limit error and return (tokens_sent, context_limit).

    Convenience wrapper around parse_context_limit_error() that handles
    the common pattern of falling back to a default limit when parsing fails.

    Args:
        error: The exception to parse.
        default_limit: Fallback context limit if parsing fails (default: 8192).

    Returns:
        Tuple of (tokens_sent, context_limit) where:
        - tokens_sent may be None if only the limit was found
        - context_limit is the parsed limit or default_limit if parsing failed
    """
    parsed = parse_context_limit_error(error)
    if parsed:
        return parsed
    return None, default_limit


def batch_by_token_limit(
    texts: List[str], token_limit: int
) -> Iterator[List[str]]:
    """Yield batches of texts fitting within token limit.

    Each batch will have a total token count at or below the limit.
    Individual texts that exceed the limit are yielded alone.

    Args:
        texts: List of texts to batch.
        token_limit: Maximum total tokens per batch.

    Yields:
        Lists of texts, each batch within token_limit.
    """
    if not texts:
        return

    batch: List[str] = []
    current_tokens = 0

    for text in texts:
        text_tokens = token_count(text)

        # Single text exceeds limit - yield alone
        if text_tokens > token_limit:
            if batch:
                yield batch
                batch, current_tokens = [], 0
            yield [text]
            continue

        # Adding would exceed limit - yield current batch first
        if current_tokens + text_tokens > token_limit and batch:
            yield batch
            batch, current_tokens = [], 0

        batch.append(text)
        current_tokens += text_tokens

    if batch:
        yield batch


def calculate_batch_token_limit(
    total_tokens: int, context_limit: int, num_texts: int
) -> int:
    """Calculate optimal batch token limit based on error info.

    Args:
        total_tokens: Total tokens that caused the error (or estimate).
        context_limit: Model's context window limit.
        num_texts: Number of texts being embedded.

    Returns:
        Token limit per batch with safety buffer.
    """
    safe_limit = int(context_limit * SAFETY_BUFFER)

    if total_tokens and total_tokens > safe_limit:
        num_batches = (total_tokens // safe_limit) + 1
        tokens_per_batch = total_tokens // num_batches
        return int(tokens_per_batch * SAFETY_BUFFER)

    return safe_limit


OUTPUT_RATIO = 0.10


def calculate_output_buffer(context_limit: int) -> int:
    """Calculate output buffer as percentage of context window.

    Args:
        context_limit: Model's context window size.

    Returns:
        Output token limit (10% of context).
    """
    return int(context_limit * OUTPUT_RATIO)


def chunk_text_by_tokens(text: str, max_tokens: int) -> List[str]:
    """Split text into chunks that fit within a token limit.

    Uses hierarchical splitting: first by paragraphs, then by sentences,
    then by words if necessary.

    Args:
        text: The text to chunk.
        max_tokens: Maximum tokens per chunk.

    Returns:
        List of text chunks, each within the token limit.
    """
    if not text:
        return []

    # If text fits, return as-is
    if token_count(text) <= max_tokens:
        return [text]

    chunks = []

    # Try splitting by paragraphs first
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        chunks = _merge_splits(paragraphs, max_tokens)
        if chunks:
            return chunks

    # Try splitting by sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        chunks = _merge_splits(sentences, max_tokens)
        if chunks:
            return chunks

    # Fall back to splitting by words
    words = text.split()
    chunks = _merge_splits(words, max_tokens, separator=" ")
    if chunks:
        return chunks

    # Last resort: hard split by character count estimate
    # Rough estimate: 1 token ≈ 4 characters
    char_limit = max_tokens * 4
    return [text[i : i + char_limit] for i in range(0, len(text), char_limit)]


def _merge_splits(
    parts: List[str], max_tokens: int, separator: str = "\n\n"
) -> List[str]:
    """Merge split parts into chunks respecting token limits.

    Args:
        parts: List of text parts to merge.
        max_tokens: Maximum tokens per chunk.
        separator: String to join parts with.

    Returns:
        List of merged chunks.
    """
    chunks = []
    current_chunk = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # If single part exceeds limit, recurse with finer splitting
        if token_count(part) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # Try splitting this part further
            if separator == "\n\n":
                # Currently splitting by paragraphs, try sentences
                sub_chunks = _merge_splits(
                    re.split(r"(?<=[.!?])\s+", part), max_tokens, " "
                )
            elif separator == " ":
                # Already at word level, hard split
                char_limit = max_tokens * 4
                sub_chunks = [
                    part[i : i + char_limit]
                    for i in range(0, len(part), char_limit)
                ]
            else:
                sub_chunks = [part]
            chunks.extend(sub_chunks)
            continue

        candidate = (
            f"{current_chunk}{separator}{part}" if current_chunk else part
        )
        if token_count(candidate) <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
