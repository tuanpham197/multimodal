import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.domain.interfaces import VectorRepository


class ChromaVectorRepository(VectorRepository):
    def __init__(self, host: str, port: int, collection_name: str, embeddings: Embeddings):
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

    @property
    def client(self) -> chromadb.HttpClient:
        return self._client

    @property
    def vectorstore(self) -> Chroma:
        return self._vectorstore

    def add_documents(self, documents: list[Document]) -> None:
        self._vectorstore.add_documents(documents)

    def search_with_scores(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        return self._vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def delete_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
        )

    def get_count(self) -> int:
        try:
            collection = self._client.get_collection(self._collection_name)
            return collection.count()
        except Exception:
            return 0

