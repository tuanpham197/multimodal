from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image, CompositeElement


class PDFParser:
    def __init__(self, strategy: str = "hi_res", max_characters: int = 10000):
        self._strategy = strategy
        self._max_characters = max_characters

    def parse(self, file_path: str) -> dict:
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy=self._strategy,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=self._max_characters,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )
        return self._filter_elements(chunks)

    def _filter_elements(self, chunks: list) -> dict:
        images, tables = self._filter_images_and_tables(chunks)
        return {
            "texts": self._filter_texts(chunks),
            "images": images,
            "tables": tables,
        }

    def _filter_texts(self, chunks: list) -> list:
        return [chunk for chunk in chunks if isinstance(chunk, CompositeElement)]

    def _filter_images_and_tables(self, chunks: list) -> tuple[list, list]:
        images, tables = [], []
        for chunk in chunks:
            if isinstance(chunk, Image):
                images.append(chunk)
            elif isinstance(chunk, Table):
                tables.append(chunk)
            elif isinstance(chunk, CompositeElement):
                for el in chunk.metadata.orig_elements or []:
                    if isinstance(el, Image):
                        images.append(el)
                    elif isinstance(el, Table):
                        tables.append(el)
        return images, tables

