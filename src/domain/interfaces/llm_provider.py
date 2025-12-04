from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel


class LLMProvider(ABC):
    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        pass

    @abstractmethod
    def get_vision_model(self) -> BaseChatModel:
        pass

