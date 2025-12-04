from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import Container
from src.interfaces.api import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = Container()

    print("\n[STARTUP] Initializing vector store...")
    index_use_case = container.get_index_use_case()

    print(f"[STARTUP] Parsing PDF: {container.settings.pdf_path}")
    result = index_use_case.index_pdf(container.settings.pdf_path)
    print(f"[STARTUP] PDF indexing result: {result}")

    if container.settings.image_paths:
        print(f"[STARTUP] Processing {len(container.settings.image_paths)} image paths...")
        img_result = index_use_case.index_images(container.settings.image_paths)
        print(f"[STARTUP] Image indexing result: {img_result}")

    print("[STARTUP] Initialization complete!")

    yield

    print("\n[SHUTDOWN] Cleaning up...")
    Container.reset_instance()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multimodal RAG API",
        description="Clean Architecture Multimodal RAG API",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

