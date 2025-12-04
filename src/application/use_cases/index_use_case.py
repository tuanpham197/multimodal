from pathlib import Path
from src.domain.interfaces import LLMProvider
from src.application.services import SummarizationService
from src.infrastructure.parsers import ImageParser, UnifiedDocumentParser
from src.infrastructure.repositories.multivector_repository import MultiVectorRepository


class IndexUseCase:
    def __init__(
        self,
        multi_vector_repo: MultiVectorRepository,
        llm_provider: LLMProvider,
    ):
        self._repo = multi_vector_repo
        self._summarization_service = SummarizationService(llm_provider)
        self._document_parser = UnifiedDocumentParser()
        self._image_parser = ImageParser()

    def index_document(self, file_path: str) -> dict:
        ext = Path(file_path).suffix.lower()
        print(f"[DEBUG IndexUseCase] Indexing document: {file_path} (type: {ext})")

        elements = self._document_parser.parse(file_path)
        total = 0

        if elements["images"]:
            print(f"[DEBUG IndexUseCase] Found {len(elements['images'])} images")
            image_summaries = self._summarization_service.summarize_images(elements["images"])
            result = self._image_parser.chunk_summaries(elements["images"], image_summaries)
            total += self._repo.add_documents(result["images"], result["summaries"])

        if elements["texts"]:
            print(f"[DEBUG IndexUseCase] Found {len(elements['texts'])} text chunks")
            text_contents = [str(t) for t in elements["texts"]]
            text_summaries = self._summarization_service.summarize_texts(text_contents)
            total += self._repo.add_documents(text_contents, text_summaries)

        if elements["tables"]:
            print(f"[DEBUG IndexUseCase] Found {len(elements['tables'])} tables")
            table_contents = [t.metadata.text_as_html if hasattr(t.metadata, "text_as_html") else str(t) for t in elements["tables"]]
            table_summaries = self._summarization_service.summarize_texts(table_contents)
            total += self._repo.add_documents(table_contents, table_summaries)

        return self._build_result(total)

    def index_pdf(self, file_path: str) -> dict:
        return self.index_document(file_path)

    def index_pptx(self, file_path: str) -> dict:
        return self.index_document(file_path)

    def index_docx(self, file_path: str) -> dict:
        return self.index_document(file_path)

    def index_images(self, image_paths: list[str]) -> dict:
        print(f"[DEBUG IndexUseCase] Processing {len(image_paths)} image paths")
        images = self._image_parser.load_multiple(image_paths)

        print(f"[DEBUG IndexUseCase] Generating summaries...")
        image_summaries = self._summarization_service.summarize_images(images)
        print(f"[DEBUG IndexUseCase] Summaries: {image_summaries}")

        result = self._image_parser.chunk_summaries(images, image_summaries)

        with open("img.txt", "w") as f:
            for i, img_base64 in enumerate(result["images"]):
                f.write(f"=== IMAGE {i+1} ===\n")
                f.write(img_base64)
                f.write("\n\n")

        total = self._repo.add_documents(result["images"], result["summaries"])
        return self._build_result(total)

    def index_custom_data(self, data: dict, summaries: dict) -> dict:
        total = 0

        if data.get("image") and summaries.get("image"):
            total += self._repo.add_documents(data["image"], summaries["image"])

        if data.get("text") and summaries.get("text"):
            texts = [str(t) for t in data["text"]]
            total += self._repo.add_documents(texts, summaries["text"])

        if data.get("table") and summaries.get("table"):
            tables = [str(t) for t in data["table"]]
            total += self._repo.add_documents(tables, summaries["table"])

        return self._build_result(total)

    def _build_result(self, total: int) -> dict:
        return {
            "total_indexed": total,
            "vectorstore_count": self._repo.get_vectorstore_count(),
            "docstore_keys": len(self._repo.get_docstore_keys()),
        }

    def clear(self) -> None:
        self._repo.delete_collection()
