from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class IndexRequest(BaseModel):
    data: dict
    summaries: dict
    images_paths: list[str] = []


class ContextImage(BaseModel):
    data_url: str
    mime_type: str = "image/jpeg"


class ChatResponse(BaseModel):
    response: str
    context_texts: list[str]
    context_images: list[ContextImage]


class HealthResponse(BaseModel):
    status: str


class StatsResponse(BaseModel):
    collection: str
    vectorstore_count: int
    inmemory_store_keys: int


class IndexResponse(BaseModel):
    status: str
    result: dict


class ClearResponse(BaseModel):
    status: str
    message: str

