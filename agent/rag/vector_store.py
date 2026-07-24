import math
import os

from google import genai

from agent.rag.chunker import chunk_text


class VectorStore:
    """In-memory vector store using Google's embedding API and cosine similarity."""

    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for embeddings.")
        self._client = genai.Client(api_key=api_key)
        self._model = "gemini-embedding-001"
        self._chunks = []
        self._vectors = []

    def add_document(self, text: str, source: str = "unknown"):
        """Chunk a document, embed each chunk, store both."""
        chunks = chunk_text(text)
        for chunk in chunks:
            chunk["source"] = source

        texts = [c["text"] for c in chunks]
        embeddings = self._embed_batch(texts)

        self._chunks.extend(chunks)
        self._vectors.extend(embeddings)

        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Find the top_k most similar chunks to the query."""
        if not self._chunks:
            return []

        query_vec = self._embed_batch([query])[0]

        scored = []
        for i, doc_vec in enumerate(self._vectors):
            score = self._cosine_similarity(query_vec, doc_vec)
            scored.append((score, i))

        scored.sort(reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            results.append({
                **self._chunks[idx],
                "score": round(score, 4),
            })
        return results

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Google's embedding API."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
