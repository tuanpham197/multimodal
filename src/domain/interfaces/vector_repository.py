from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.documents import Document


class VectorRepository(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        pass

    @abstractmethod
    def search_with_scores(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        pass

    @abstractmethod
    def get_count(self) -> int:
        pass

