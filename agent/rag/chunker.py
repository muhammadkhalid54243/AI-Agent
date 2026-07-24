def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """Split text into overlapping chunks.

    Each chunk is a dict with 'text' and 'index' (position in the original).
    Overlap ensures we don't lose meaning at chunk boundaries — if a sentence
    spans two chunks, the overlap catches it in at least one.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append({
            "text": " ".join(chunk_words),
            "index": len(chunks),
        })
        start += chunk_size - overlap

    return chunks
