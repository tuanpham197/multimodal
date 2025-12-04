from fastapi import APIRouter, Depends, Query

from src.interfaces.api.schemas import (
    ChatRequest,
    ChatResponse,
    ContextImage,
    IndexRequest,
    IndexResponse,
    ClearResponse,
    StatsResponse,
    HealthResponse,
)
from src.application.use_cases import ChatUseCase, IndexUseCase
from src.config.container import Container

router = APIRouter()


def get_container() -> Container:
    return Container()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, container: Container = Depends(get_container)):
    chat_use_case = container.get_chat_use_case()
    result = chat_use_case.execute(request.message)

    images = [
        ContextImage(data_url=f"data:image/jpeg;base64,{img}")
        for img in result.context.images[:3]
    ]

    return ChatResponse(
        response=result.response,
        context_texts=result.context.texts[:3],
        context_images=images,
    )


@router.post("/clear", response_model=ClearResponse)
def clear_collection(container: Container = Depends(get_container)):
    index_use_case = container.get_index_use_case()
    index_use_case.clear()
    return ClearResponse(status="ok", message="Collection cleared")


@router.post("/reindex", response_model=IndexResponse)
def reindex_data(container: Container = Depends(get_container)):
    index_use_case = container.get_index_use_case()
    result = index_use_case.index_pdf(container.settings.pdf_path)
    return IndexResponse(status="ok", result=result)


@router.post("/index/document", response_model=IndexResponse)
def index_document(file_path: str = Query(...), container: Container = Depends(get_container)):
    index_use_case = container.get_index_use_case()
    result = index_use_case.index_document(file_path)
    return IndexResponse(status="ok", result=result)


@router.post("/index/pptx", response_model=IndexResponse)
def index_pptx(file_path: str = Query(...), container: Container = Depends(get_container)):
    index_use_case = container.get_index_use_case()
    result = index_use_case.index_pptx(file_path)
    return IndexResponse(status="ok", result=result)


@router.post("/index/docx", response_model=IndexResponse)
def index_docx(file_path: str = Query(...), container: Container = Depends(get_container)):
    index_use_case = container.get_index_use_case()
    result = index_use_case.index_docx(file_path)
    return IndexResponse(status="ok", result=result)


@router.post("/index", response_model=IndexResponse)
def index_custom_data(request: IndexRequest, container: Container = Depends(get_container)):
    index_use_case = container.get_index_use_case()

    data = request.data.copy()
    summaries = request.summaries.copy()

    if request.images_paths:
        images_result = index_use_case.index_images(request.images_paths)
        if "image" not in data:
            data["image"] = []
        if "image" not in summaries:
            summaries["image"] = []

    result = index_use_case.index_custom_data(data, summaries)
    return IndexResponse(status="ok", result=result)


@router.get("/stats", response_model=StatsResponse)
def get_stats(container: Container = Depends(get_container)):
    repo = container.get_multi_vector_repository()

    return StatsResponse(
        collection=container.settings.collection_name,
        vectorstore_count=repo.get_vectorstore_count(),
        inmemory_store_keys=len(repo.get_docstore_keys()),
    )


@router.get("/debug")
def debug_data(container: Container = Depends(get_container)):
    repo = container.get_multi_vector_repository()

    keys = repo.get_docstore_keys()
    sample_data = []
    for key in keys[:5]:
        data = repo.get_raw_by_id(key)
        if data:
            preview = data[:200] if isinstance(data, str) else str(data)[:200]
            sample_data.append({"key": key, "preview": preview, "length": len(data) if hasattr(data, '__len__') else 0})

    return {
        "vectorstore_count": repo.get_vectorstore_count(),
        "docstore_keys_count": len(keys),
        "sample_keys": keys[:10],
        "sample_data": sample_data,
    }


@router.get("/debug/search")
def debug_search(q: str, container: Container = Depends(get_container)):
    repo = container.get_multi_vector_repository()

    results = repo.search_with_scores(q, k=3)

    search_results = []
    for doc, score in results:
        doc_id = doc.metadata.get("doc_id")
        raw_data = repo.get_raw_by_id(doc_id) if doc_id else None
        search_results.append({
            "score": score,
            "summary": doc.page_content[:300],
            "doc_id": doc_id,
            "raw_data_found": raw_data is not None,
            "raw_data_preview": (raw_data[:200] if isinstance(raw_data, str) else str(raw_data)[:200]) if raw_data else None,
        })

    return {
        "query": q,
        "results_count": len(results),
        "results": search_results,
    }


@router.get("/debug/retriever")
def debug_retriever(q: str, container: Container = Depends(get_container)):
    repo = container.get_multi_vector_repository()

    docs = repo.search(q, k=3)

    results = []
    for i, doc in enumerate(docs):
        preview = doc[:200] if isinstance(doc, str) else str(doc)[:200]
        results.append({
            "index": i,
            "length": len(doc) if hasattr(doc, '__len__') else 0,
            "preview": preview,
        })

    return {
        "query": q,
        "results_count": len(docs),
        "results": results,
    }
