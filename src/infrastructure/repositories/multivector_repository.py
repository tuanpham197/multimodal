import uuid
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever


class MultiVectorRepository:
    ID_KEY = "doc_id"

    def __init__(self, host: str, port: int, collection_name: str, embeddings: Embeddings):
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._store = InMemoryStore()
        self._init_vectorstore()

    def _init_vectorstore(self):
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
        )
        self._retriever = MultiVectorRetriever(
            vectorstore=self._vectorstore,
            docstore=self._store,
            id_key=self.ID_KEY,
        )

    @property
    def store(self) -> InMemoryStore:
        return self._store

    @property
    def retriever(self) -> MultiVectorRetriever:
        return self._retriever

    def add_documents(self, items: list, summaries: list) -> int:
        doc_ids = [str(uuid.uuid4()) for _ in items]
        
        summary_docs = [
            Document(page_content=summary, metadata={self.ID_KEY: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]
        
        print(f"[DEBUG MultiVectorRepo] Adding {len(summary_docs)} summaries to vectorstore")
        self._retriever.vectorstore.add_documents(summary_docs)
        
        print(f"[DEBUG MultiVectorRepo] Adding {len(items)} items to docstore")
        self._retriever.docstore.mset(list(zip(doc_ids, items)))
        
        return len(doc_ids)

    def search(self, query: str, k: int = 3) -> list:
        print(f"[DEBUG MultiVectorRepo] Searching: {query}")
        docs = self._retriever.invoke(query, k=k)
        print(f"[DEBUG MultiVectorRepo] Found {len(docs)} documents")
        return docs

    def search_with_scores(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        return self._vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def get_raw_by_id(self, doc_id: str):
        result = self._store.mget([doc_id])
        return result[0] if result else None

    def delete_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
            print(f"[DEBUG MultiVectorRepo] Deleted collection: {self._collection_name}")
        except Exception:
            pass
        self._store = InMemoryStore()
        self._init_vectorstore()

    def get_vectorstore_count(self) -> int:
        try:
            collection = self._client.get_collection(self._collection_name)
            return collection.count()
        except Exception:
            return 0

    def get_docstore_keys(self) -> list[str]:
        return list(self._store.yield_keys())

