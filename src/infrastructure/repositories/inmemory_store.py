from typing import Any, Optional
from langchain_core.stores import InMemoryStore as LangchainInMemoryStore
from src.domain.interfaces import DocumentStore


class InMemoryDocumentStore(DocumentStore):
    def __init__(self):
        self._store = LangchainInMemoryStore()

    @property
    def store(self) -> LangchainInMemoryStore:
        return self._store

    def set(self, key: str, value: Any) -> None:
        self._store.mset([(key, value)])

    def get(self, key: str) -> Optional[Any]:
        results = self._store.mget([key])
        return results[0] if results else None

    def mset(self, key_value_pairs: list[tuple[str, Any]]) -> None:
        self._store.mset(key_value_pairs)

    def mget(self, keys: list[str]) -> list[Optional[Any]]:
        return self._store.mget(keys)

    def clear(self) -> None:
        self._store = LangchainInMemoryStore()

    def keys(self) -> list[str]:
        return list(self._store.yield_keys())

