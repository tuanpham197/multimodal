import ollama
from langchain_core.embeddings import Embeddings
from src.domain.interfaces import EmbeddingProvider


class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text", max_length: int = 2000, host: str = "http://localhost:11434"):
        self.model = model
        self.max_length = max_length
        self.client = ollama.Client(host=host)

    def _truncate(self, text: str) -> str:
        if len(text) > self.max_length:
            return text[:self.max_length]
        return text

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            truncated_text = self._truncate(text)
            response = self.client.embed(model=self.model, input=truncated_text)
            embeddings.append(response["embeddings"][0])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        truncated_text = self._truncate(text)
        response = self.client.embed(model=self.model, input=truncated_text)
        return response["embeddings"][0]


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "nomic-embed-text", max_length: int = 2000, host: str = "http://localhost:11434"):
        self._embeddings = OllamaEmbeddings(model=model, max_length=max_length, host=host)

    def get_embeddings(self) -> Embeddings:
        return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

