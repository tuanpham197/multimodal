from langchain_core.output_parsers import StrOutputParser

from src.domain.entities import ChatContext, ChatResult
from src.domain.interfaces import LLMProvider
from src.application.services import PromptBuilder, DocumentParser
from src.infrastructure.repositories.multivector_repository import MultiVectorRepository


class ChatUseCase:
    ID_KEY = "doc_id"

    def __init__(
        self,
        llm_provider: LLMProvider,
        multi_vector_repo: MultiVectorRepository,
    ):
        self._llm_provider = llm_provider
        self._repo = multi_vector_repo
        self._prompt_builder = PromptBuilder()
        self._document_parser = DocumentParser()

    def execute(self, message: str) -> ChatResult:
        relevant_docs = self._retrieve_documents(message)
        print(f"[DEBUG ChatUseCase] Message: {message}")
        print(f"[DEBUG ChatUseCase] Retrieved {len(relevant_docs)} documents")
        for i, doc in enumerate(relevant_docs):
            preview = doc if isinstance(doc, str) else str(doc)
            print(f"[DEBUG ChatUseCase] Doc {i}: len={len(doc) if hasattr(doc, '__len__') else 'N/A'}, preview={preview}...")

        parsed = self._document_parser.parse_documents(relevant_docs)
        context = ChatContext(texts=parsed["texts"], images=parsed["images"])

        prompt = self._prompt_builder.build(context, message)
        model = self._llm_provider.get_vision_model()
        chain = prompt | model | StrOutputParser()

        response = chain.invoke({})

        return ChatResult(response=response, context=context)

    def _retrieve_documents(self, query: str, k: int = 3) -> list:
        print(f"\n[DEBUG _retrieve] Query: {query}")
        print(f"[DEBUG _retrieve] VectorStore count: {self._repo.get_vectorstore_count()}")
        print(f"[DEBUG _retrieve] DocStore keys: {len(self._repo.get_docstore_keys())}")

        docs = self._repo.search(query, k=k)
        print(f"[DEBUG _retrieve] MultiVectorRetriever returned {len(docs)} docs")

        return docs
