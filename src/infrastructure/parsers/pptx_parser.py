import base64
from pathlib import Path
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from unstructured.partition.pptx import partition_pptx
from unstructured.documents.elements import Table, Image, CompositeElement


class PptxParser:
    def extract_images(self, file_path: str) -> list[str]:
        prs = Presentation(file_path)
        images_base64 = []

        for slide_num, slide in enumerate(prs.slides, 1):
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_bytes = shape.image.blob
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    images_base64.append(image_base64)
                    print(f"[DEBUG PptxParser] Slide {slide_num}: Found image, size={len(image_bytes)} bytes")

                if hasattr(shape, "shapes"):
                    for sub_shape in shape.shapes:
                        if sub_shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                            image_bytes = sub_shape.image.blob
                            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                            images_base64.append(image_base64)
                            print(f"[DEBUG PptxParser] Slide {slide_num}: Found grouped image")

        print(f"[DEBUG PptxParser] Total images extracted: {len(images_base64)}")
        return images_base64

    def extract_text_and_tables(self, file_path: str) -> dict:
        chunks = partition_pptx(filename=file_path)
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

