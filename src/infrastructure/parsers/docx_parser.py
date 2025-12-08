import base64
import re
from pathlib import Path
import docx2txt
from unstructured.partition.docx import partition_docx
from unstructured.documents.elements import Table, Title
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocxParser:
    def __init__(
        self,
        temp_dir: str = "./temp_images",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self._temp_dir = Path(temp_dir)
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", ":", ";", ",", " ", ""],
        )

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
                    print(f"[DEBUG DocxParser] Found image: {image_path.name}")

        print(f"[DEBUG DocxParser] Total images extracted: {len(images_base64)}")
        return images_base64

    def extract_text_and_tables(self, file_path: str) -> dict:
        elements = partition_docx(filename=file_path)
        print(f"\n[DEBUG DocxParser] Raw elements: {len(elements)}")

        sections = self._group_by_sections(elements)
        print(f"[DEBUG DocxParser] Grouped into {len(sections)} sections")

        all_chunks = []
        tables = []

        for section in sections:
            if section["type"] == "table":
                tables.append(section["content"])
            else:
                section_text = section["content"]
                if len(section_text) > self._text_splitter._chunk_size:
                    chunks = self._text_splitter.split_text(section_text)
                    all_chunks.extend(chunks)
                elif len(section_text) >= 50:
                    all_chunks.append(section_text)

        print(f"\n[DEBUG DocxParser] Final chunks: {len(all_chunks)} texts, {len(tables)} tables")
        for i, chunk in enumerate(all_chunks[:5]):
            print(f"[DEBUG DocxParser] Chunk[{i}] ({len(chunk)} chars): {chunk[:150]}...")

        with open("chunk.txt", "w", encoding="utf-8") as f:
            for i, chunk in enumerate(all_chunks):
                f.write(f"=== CHUNK {i+1} ({len(chunk)} chars) ===\n")
                f.write(chunk)
                f.write("\n\n" + "="*50 + "\n\n")
            f.write(f"\n\n=== TABLES ({len(tables)}) ===\n\n")
            for i, table in enumerate(tables):
                f.write(f"--- TABLE {i+1} ---\n")
                f.write(str(table))
                f.write("\n\n")
        print(f"[DEBUG DocxParser] Written all chunks to chunk.txt")

        return {"texts": all_chunks, "tables": tables}

    def _group_by_sections(self, elements: list) -> list[dict]:
        sections = []
        current_section = []

        for element in elements:
            if isinstance(element, Table):
                if current_section:
                    sections.append({
                        "type": "text",
                        "content": self._merge_texts(current_section)
                    })
                    current_section = []
                table_text = element.metadata.text_as_html if hasattr(element.metadata, 'text_as_html') else str(element)
                sections.append({"type": "table", "content": table_text})
                continue

            text = element.text if hasattr(element, 'text') else str(element)
            if not text or not text.strip():
                continue

            is_major_title = isinstance(element, Title) and self._is_major_section(text)

            if is_major_title and current_section:
                sections.append({
                    "type": "text",
                    "content": self._merge_texts(current_section)
                })
                current_section = []

            current_section.append(text)

        if current_section:
            sections.append({
                "type": "text",
                "content": self._merge_texts(current_section)
            })

        return sections

    def _is_major_section(self, text: str) -> bool:
        text = text.strip()
        if re.match(r'^[IVXLCDM]+\.\s+', text):
            return True
        if re.match(r'^[A-Z]\.\s+', text):
            return True
        if re.match(r'^(Phần|Chương|Mục)\s+\d+', text, re.IGNORECASE):
            return True
        return False

    def _merge_texts(self, texts: list[str]) -> str:
        return "\n".join(t.strip() for t in texts if t.strip())

    def parse(self, file_path: str) -> dict:
        images = self.extract_images(file_path)
        elements = self.extract_text_and_tables(file_path)

        print(f"\n[DEBUG DocxParser] === PARSE RESULT ===")
        print(f"[DEBUG DocxParser] Images: {len(images)}")
        print(f"[DEBUG DocxParser] Texts: {len(elements['texts'])}")
        print(f"[DEBUG DocxParser] Tables: {len(elements['tables'])}")

        return {
            "images": images,
            "texts": elements["texts"],
            "tables": elements["tables"],
        }
