import base64
from pathlib import Path
import docx2txt
from unstructured.partition.docx import partition_docx
from unstructured.documents.elements import Table, Image, CompositeElement


class DocxParser:
    def __init__(self, temp_dir: str = "./temp_images"):
        self._temp_dir = Path(temp_dir)

    def extract_images(self, file_path: str) -> list[str]:
        output_dir = self._temp_dir / Path(file_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            docx2txt.process(file_path, str(output_dir))
        except Exception as e:
            print(f"[DEBUG DocxParser] Error processing DOCX: {e}")
            return []

        images_base64 = []
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]

        for image_path in output_dir.glob("*"):
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                with open(image_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                    images_base64.append(img_base64)
                    print(f"[DEBUG DocxParser] Found image: {image_path.name}, size={len(img_base64)} chars")

        print(f"[DEBUG DocxParser] Total images extracted: {len(images_base64)}")
        return images_base64

    def extract_text_and_tables(self, file_path: str) -> dict:
        chunks = partition_docx(filename=file_path)
        return self._filter_elements(chunks)

    def parse(self, file_path: str) -> dict:
        images = self.extract_images(file_path)
        elements = self.extract_text_and_tables(file_path)

        return {
            "images": images,
            "texts": elements["texts"],
            "tables": elements["tables"],
        }

    def _filter_elements(self, chunks: list) -> dict:
        texts = []
        tables = []

        for chunk in chunks:
            if isinstance(chunk, Table):
                tables.append(chunk)
            elif isinstance(chunk, CompositeElement):
                texts.append(chunk)
                for el in chunk.metadata.orig_elements or []:
                    if isinstance(el, Table):
                        tables.append(el)

        return {"texts": texts, "tables": tables}

