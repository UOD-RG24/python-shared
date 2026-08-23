from __future__ import annotations

from hashlib import sha256
from typing import BinaryIO, Final

DEFAULT_CHUNK_SIZE: Final[int] = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def sha256_stream(
    stream: BinaryIO,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    rewind: bool = True,
) -> tuple[str, int]:
    """Hash a binary stream without retaining it in memory.

    The stream is read from its current position and, when seekable, restored to
    that position by default.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    original_position: int | None = None
    if rewind and stream.seekable():
        original_position = stream.tell()

    digest = sha256()
    byte_length = 0
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
        byte_length += len(chunk)

    if original_position is not None:
        stream.seek(original_position)
    return digest.hexdigest(), byte_length
