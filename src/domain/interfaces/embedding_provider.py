from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings


class EmbeddingProvider(ABC):
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass

