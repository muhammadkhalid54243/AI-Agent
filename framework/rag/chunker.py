"""Split text into overlapping word-window chunks.

Overlap keeps meaning intact at chunk boundaries — a sentence spanning two chunks
is captured whole in at least one.
"""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += max(1, chunk_size - overlap)
    return chunks
