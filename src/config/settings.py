import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    chroma_host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", "localhost"))
    chroma_port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8001")))

    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_embed_model: str = "nomic-embed-text"
    ollama_chat_model: str = "llama3"
    ollama_vision_model: str = "llava"

    google_chat_model: str = "gemini-2.0-flash-lite"
    google_vision_model: str = "gemini-2.0-flash"

    collection_name: str = "multi_modal_rag"
    pdf_path: str = "./content/attention.pdf"
    image_paths: list[str] = field(default_factory=lambda: ["./content/images/image.png"])

    llm_provider_type: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "google"))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()

