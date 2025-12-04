import base64
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ImageParser:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "*", " ", ""]
        )

    def load_as_base64(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def load_multiple(self, file_paths: list[str]) -> list[str]:
        return [self.load_as_base64(path) for path in file_paths]

    def chunk_summaries(self, images: list[str], summaries: list[str]) -> dict:
        chunked_images = []
        chunked_summaries = []

        for img, summary in zip(images, summaries):
            chunks = self._text_splitter.split_text(summary)
            print(f"[DEBUG ImageParser] Split summary into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"[DEBUG ImageParser] Chunk {i}: {len(chunk)} chars - {chunk[:100]}...")
                chunked_images.append(img)
                chunked_summaries.append(chunk)

        print(f"[DEBUG ImageParser] Total: {len(images)} images -> {len(chunked_images)} chunks")

        return {
            "images": chunked_images,
            "summaries": chunked_summaries,
            "original_count": len(images),
            "chunk_count": len(chunked_images),
        }
