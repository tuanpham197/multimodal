from typing import Optional
from src.config.settings import Settings
from src.domain.interfaces import LLMProvider
from src.infrastructure import (
    GoogleLLMProvider,
    OllamaLLMProvider,
    OllamaEmbeddingProvider,
)
from src.infrastructure.repositories.multivector_repository import MultiVectorRepository
from src.application.use_cases import ChatUseCase, IndexUseCase


class Container:
    _instance: Optional["Container"] = None
    _settings: Optional[Settings] = None
    _indexing_llm_provider: Optional[LLMProvider] = None
    _chat_llm_provider: Optional[LLMProvider] = None
    _embedding_provider: Optional[OllamaEmbeddingProvider] = None
    _multi_vector_repo: Optional[MultiVectorRepository] = None

    def __new__(cls) -> "Container":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = Settings.from_env()
        return self._settings

    def get_indexing_llm_provider(self) -> LLMProvider:
        if self._indexing_llm_provider is None:
            self._indexing_llm_provider = OllamaLLMProvider(
                model=self.settings.ollama_chat_model,
                vision_model=self.settings.ollama_vision_model,
            )
        return self._indexing_llm_provider

    def get_chat_llm_provider(self) -> LLMProvider:
        if self._chat_llm_provider is None:
            if self.settings.google_api_key:
                self._chat_llm_provider = GoogleLLMProvider(
                    api_key=self.settings.google_api_key,
                    model=self.settings.google_chat_model,
                )
            else:
                print("[WARN] GOOGLE_API_KEY not set, using Ollama for chat")
                self._chat_llm_provider = OllamaLLMProvider(
                    model=self.settings.ollama_chat_model,
                    vision_model=self.settings.ollama_vision_model,
                )
        return self._chat_llm_provider

    def get_embedding_provider(self) -> OllamaEmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = OllamaEmbeddingProvider(
                model=self.settings.ollama_embed_model,
                host=self.settings.ollama_host,
            )
        return self._embedding_provider

    def get_multi_vector_repository(self) -> MultiVectorRepository:
        if self._multi_vector_repo is None:
            embedding_provider = self.get_embedding_provider()
            self._multi_vector_repo = MultiVectorRepository(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port,
                collection_name=self.settings.collection_name,
                embeddings=embedding_provider.get_embeddings(),
            )
        return self._multi_vector_repo

    def get_chat_use_case(self) -> ChatUseCase:
        return ChatUseCase(
            llm_provider=self.get_chat_llm_provider(),
            multi_vector_repo=self.get_multi_vector_repository(),
        )

    def get_index_use_case(self) -> IndexUseCase:
        return IndexUseCase(
            multi_vector_repo=self.get_multi_vector_repository(),
            llm_provider=self.get_indexing_llm_provider(),
        )

    def reset(self) -> None:
        self._indexing_llm_provider = None
        self._chat_llm_provider = None
        self._embedding_provider = None
        self._multi_vector_repo = None

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None
        cls._settings = None
        cls._indexing_llm_provider = None
        cls._chat_llm_provider = None
        cls._embedding_provider = None
        cls._multi_vector_repo = None
