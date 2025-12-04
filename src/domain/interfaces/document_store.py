from abc import ABC, abstractmethod
from typing import Any, Optional


class DocumentStore(ABC):
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def mset(self, key_value_pairs: list[tuple[str, Any]]) -> None:
        pass

    @abstractmethod
    def mget(self, keys: list[str]) -> list[Optional[Any]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def keys(self) -> list[str]:
        pass

