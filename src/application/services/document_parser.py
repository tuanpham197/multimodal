import re
from base64 import b64decode


class DocumentParser:
    @staticmethod
    def is_valid_base64_image(content: str) -> bool:
        if len(content) < 500:
            return False
        if not re.match(r'^[A-Za-z0-9+/=]+$', content):
            return False
        try:
            decoded = b64decode(content, validate=True)
            return len(decoded) > 100
        except Exception:
            return False

    def parse_documents(self, docs: list) -> dict:
        images = []
        texts = []

        for doc in docs:
            content = doc.page_content if hasattr(doc, 'page_content') else doc
            if self.is_valid_base64_image(content):
                images.append(content)
            else:
                texts.append(content)

        return {"images": images, "texts": texts}

