from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from src.domain.interfaces import LLMProvider


class GoogleLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite", temperature: float = 0):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature

    def get_chat_model(self) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            model=self._model,
            temperature=self._temperature,
            google_api_key=self._api_key,
        )

    def get_vision_model(self) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=self._temperature,
            max_tokens=65536,
            timeout=60,
            max_retries=2,
            google_api_key=self._api_key,
        )
