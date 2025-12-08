tiêu thụ trứng bình quân đầu người năm 2024 của trung quốc



## Architecture Structure
```
src/
├── domain/                    # Core business logic (no dependencies)
│   ├── entities/              # Business objects
│   │   └── document.py        # DocumentEntity, ChatContext, ChatResult
│   └── interfaces/            # Ports/Abstractions
│       ├── llm_provider.py    # LLM abstraction
│       ├── embedding_provider.py
│       ├── vector_repository.py
│       └── document_store.py
│
├── application/               # Use cases & services
│   ├── services/              # Application services
│   │   ├── prompt_builder.py  # Builder Pattern for prompts
│   │   ├── summarization_service.py
│   │   └── document_parser.py
│   └── use_cases/             # Business use cases
│       ├── chat_use_case.py   # Chat functionality
│       └── index_use_case.py  # Indexing functionality
│
├── infrastructure/            # External implementations
│   ├── llm/                   # Strategy Pattern for LLM
│   │   ├── google_provider.py
│   │   └── ollama_provider.py
│   ├── embeddings/
│   │   └── ollama_embeddings.py
│   ├── repositories/          # Repository Pattern
│   │   ├── chroma_repository.py
│   │   └── inmemory_store.py
│   └── parsers/
│       ├── pdf_parser.py
│       └── image_parser.py
│
├── interfaces/                # API Layer
│   └── api/
│       ├── routes.py          # FastAPI endpoints
│       └── schemas.py         # Request/Response models
│
├── config/                    # Configuration
│   ├── settings.py            # Environment settings
│   └── container.py           # Dependency Injection (Singleton)
│
├── data/                      # Mock data separated
│   └── mock_data.py
│
└── app.py                     # Application entry point
```

### 1. Run api
```
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### 2. Google cloud login
```
gcloud auth application-default login
```


# 1. Clear ChromaDB
curl -X POST http://localhost:8000/clear

# 2. Check đã clear chưa
curl http://localhost:8000/stats
# Phải thấy: vectorstore_count: 0, inmemory_store_keys: 0