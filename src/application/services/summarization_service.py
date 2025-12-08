import os
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from src.domain.interfaces import LLMProvider
from src.application.services.prompt_builder import ImageSummaryPromptBuilder


class SummarizationService:
    def __init__(self, llm_provider: LLMProvider):
        self._llm_provider = llm_provider
        self._prompt_builder = ImageSummaryPromptBuilder()

    def summarize_images(self, images: list[str]) -> list[str]:
        if not images:
            return []

        prompt = self._prompt_builder.build()
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=65536,
            timeout=60,
            max_retries=1,
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )
        chain = prompt | model | StrOutputParser()

        return chain.batch(images)

    def summarize_texts(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(
            """You are an assistant tasked with summarizing text.
Give a concise summary of the text.
Respond only with the summary, no additional comment.

Text chunk: {element}"""
        )

        model = self._llm_provider.get_chat_model()
        chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        return chain.batch(texts, {"max_concurrency": 3})
