from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DocumentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class DocumentEntity:
    id: str
    content: str
    doc_type: DocumentType
    metadata: dict = field(default_factory=dict)
    summary: Optional[str] = None


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatContext:
    texts: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


@dataclass
class ChatResult:
    response: str
    context: ChatContext

