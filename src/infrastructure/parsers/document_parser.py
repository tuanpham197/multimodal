from pathlib import Path
from src.infrastructure.parsers.pdf_parser import PDFParser
from src.infrastructure.parsers.pptx_parser import PptxParser
from src.infrastructure.parsers.docx_parser import DocxParser


class UnifiedDocumentParser:
    def __init__(self):
        self._pdf_parser = PDFParser()
        self._pptx_parser = PptxParser()
        self._docx_parser = DocxParser()

    def parse(self, file_path: str) -> dict:
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            return self._parse_pdf(file_path)
        elif ext == ".pptx":
            return self._pptx_parser.parse(file_path)
        elif ext == ".docx":
            return self._docx_parser.parse(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _parse_pdf(self, file_path: str) -> dict:
        elements = self._pdf_parser.parse(file_path)
        images = [img.metadata.image_base64 for img in elements["images"] if hasattr(img.metadata, "image_base64")]
        return {
            "images": images,
            "texts": elements["texts"],
            "tables": elements["tables"],
        }

    def get_supported_extensions(self) -> list[str]:
        return [".pdf", ".pptx", ".docx"]

