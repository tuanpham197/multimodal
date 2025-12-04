from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from src.domain.interfaces import LLMProvider


class OllamaLLMProvider(LLMProvider):
    def __init__(self, model: str = "llama3", vision_model: str = "llava", temperature: float = 0):
        self._model = model
        self._vision_model = vision_model
        self._temperature = temperature

    def get_chat_model(self) -> BaseChatModel:
        return ChatOllama(model=self._model, temperature=self._temperature)

    def get_vision_model(self) -> BaseChatModel:
        return ChatOllama(model=self._vision_model, temperature=self._temperature)

