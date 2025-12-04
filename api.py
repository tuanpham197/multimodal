from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from unstructured.partition.pdf import partition_pdf
from base64 import b64decode
import chromadb
import ollama
import os
import re

from main import (
    store_vector,
    OllamaEmbeddingsCustom,
    filter_elements,
    summary_image,
    extract_and_chunk_images,
)

COLLECTION_NAME = "multi_modal_rag"
ID_KEY = "doc_id"
PDF_PATH = "./content/attention.pdf"
IMAGE_PATHS = ["./content/images/image.png"]
store = InMemoryStore()

MOCK_TEXTS = [
    "The quiet morning breeze drifted through the open window, carrying the scent of fresh leaves and sunlight. It was a moment of calm that reminded everyone to slow down and appreciate simple things.",
    "Under the glowing streetlights, the city moved with familiar rhythm. People hurried past one another, yet each carried stories, dreams, and worries unseen by those sharing the same narrow sidewalks.",
    "He stared at the old photograph, wondering how time had flowed so quickly. Memories once vivid grew soft, yet the feelings remained warm, like a gentle reminder of moments that shaped his journey.",
    "A distant echo rolled across the hills as twilight settled in. The warm colors of the sunset faded slowly, leaving behind a sky filled with calm blue shadows and a quiet promise of tomorrow.",
    "She opened the dusty book and found a pressed flower between the pages. Though fragile, it held traces of laughter and days long gone, carrying a silent message of love and remembrance.",
    "Rain tapped lightly on the rooftop, creating a steady rhythm that soothed the restless night. Each drop seemed to whisper small secrets about nature's patience and its gentle way of healing.",
    "The market buzzed with voices, bright colors, and warm scents. Fresh fruits lined wooden tables, and the chatter of vendors blended into a lively melody that made the place feel alive.",
    "He walked along the beach, leaving footprints that the waves quickly erased. The ocean moved endlessly, reminding him that life changes constantly, yet always brings new tides and possibilities.",
    "Soft lanterns flickered beside the path, guiding travelers through the quiet evening. Their warm glow offered comfort, making even the longest journey feel a little less lonely and uncertain.",
    "In the small workshop, tools hung neatly on the walls. The craftsman shaped each piece with care, knowing that patience and dedication could turn simple materials into something lasting and meaningful."
]

def parse_pdf_and_get_data():
    print(f"\n[STARTUP] Parsing PDF: {PDF_PATH}")
    
    chunks = partition_pdf(
        filename=PDF_PATH,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    
    elements = filter_elements(chunks)
    
    print(f"[STARTUP] Total chunks: {len(chunks)}")
    print(f"[STARTUP] Texts: {len(elements['texts'])}")
    print(f"[STARTUP] Images: {len(elements['images'])}")
    print(f"[STARTUP] Tables: {len(elements['tables'])}")
    
    images = [img.metadata.image_base64 for img in elements['images']]
    
    print("[STARTUP] Generating image summaries...")
    image_summaries = summary_image(images)
    print(f"[STARTUP] Generated {len(image_summaries)} image summaries")
    
    all_images = images + MOCK_TEXTS
    all_summaries = image_summaries + MOCK_TEXTS
    
    print(f"[STARTUP] Adding {len(MOCK_TEXTS)} mock texts")
    print(f"[STARTUP] Total data items: {len(all_images)}")
    
    data = {
        'text': [],
        'table': [],
        'image': all_images,
    }
    
    summaries = {
        'text': [],
        'table': [],
        'image': all_summaries,
    }
    
    return data, summaries

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    print("\n[STARTUP] Initializing vector store...")
    store = InMemoryStore()
    
    data, summaries = parse_pdf_and_get_data()
    
    if IMAGE_PATHS:
        print(f"[STARTUP] Processing {len(IMAGE_PATHS)} image paths...")
        result_image = extract_and_chunk_images(IMAGE_PATHS, chunk_size=500, chunk_overlap=50)
        
        if "image" not in data:
            data["image"] = []
        if "image" not in summaries:
            summaries["image"] = []
            
        data["image"].extend(result_image["images"])
        summaries["image"].extend(result_image["summaries"])
        print(result_image["summaries"])
        
        print(f"[STARTUP] Added {result_image['chunk_count']} chunks from images")
    
    result = store_vector(store, data, summaries, COLLECTION_NAME)
    print(f"[STARTUP] Indexing result: {result}")
    yield
    print("\n[SHUTDOWN] Cleaning up...")

app = FastAPI(title="Multimodal RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def get_chroma_client():
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
    return chromadb.HttpClient(host=chroma_host, port=chroma_port)

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

def parse_docs(docs):
    print(f"\n[DEBUG parse_docs] Received {len(docs)} documents")
    b64 = []
    text = []
    for i, doc in enumerate(docs):
        content = doc.page_content if hasattr(doc, 'page_content') else doc
        print(f"[DEBUG parse_docs] Doc {i}: type={type(doc).__name__}, len={len(content)}, preview={content[:80]}...")
        if is_valid_base64_image(content):
            b64.append(content)
            print(f"[DEBUG parse_docs] Doc {i}: Detected as BASE64 IMAGE")
        else:
            text.append(content)
            print(f"[DEBUG parse_docs] Doc {i}: Detected as TEXT")
    print(f"[DEBUG parse_docs] Result: {len(b64)} images, {len(text)} texts\n")
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    
    print(f"\n[DEBUG build_prompt] Images: {len(docs_by_type['images'])}, Texts: {len(docs_by_type['texts'])}")
    
    has_context = docs_by_type["texts"] or docs_by_type["images"]
    
    if has_context:
        context_text = "\n".join(docs_by_type["texts"]) if docs_by_type["texts"] else ""
        prompt_template = f"""
        Answer the question based only on the following context, which can include text, tables, and the below image.
        System Prompt: Expert Document Analysis and Information Retrieval Agent
        You are an expert Document Analysis and Information Retrieval Agent. Your primary function is to provide the most accurate, comprehensive, and contextually relevant answers to user queries based only on the provided document context.
        1. Role and Goal
         - Role: Expert Document Analyst, Information Synthesizer, and Multimodal Data Interpreter.
         - Goal: To extract, synthesize, and present information from diverse document formats (PDF, DOCX, PPTX) to answer the user's question with high fidelity and precision.
        2. Input Context
        You will receive a user query and a set of document chunks (text, tables, image descriptions, chart data) retrieved from the source documents.
        Crucially, some chunks may be multimodal:
        Text Chunks: Standard paragraphs, lists, and tables.
        Image/Chart Chunks: These will be provided as a textual description, a caption, or the raw data extracted from the visual element (e.g., "Image: A bar chart showing sales figures. Q1: $10M, Q2: $12M...").
        3. Core Instructions (Textual Data)
        Accuracy First: Base your answer strictly on the provided context. Do not introduce external knowledge, speculation, or personal opinions.
        Synthesis: Do not simply concatenate chunks. Synthesize the information into a coherent, well-structured, and easy-to-read response.
        Handling Ambiguity: If the context is insufficient, contradictory, or does not contain the answer, state clearly and politely that the answer cannot be fully determined from the provided documents.
        4. Multimodal Instructions (Images and Charts)
        Integrate Visual Data: Treat information derived from images and charts (provided as text/data in the chunks) with the same importance as standard text.
        Prioritize Data: When a chart or table chunk provides specific numerical data, use that data directly in your answer to provide a precise response.
        Describe Visual Evidence: If the answer relies heavily on an image or chart, mention the type of visual evidence (e.g., "According to the 'Quarterly Sales' bar chart...") to justify your claim.
        Focus on Content: If an image is described as purely decorative, you may disregard it. Focus only on images and charts that convey substantive information.
        5. Output Format and Constraints
        Clarity and Tone: Maintain a professional, objective, and authoritative tone.
        Direct Answer: Start your response with a direct, concise answer to the user's question, followed by the detailed explanation and supporting evidence.
        Citation: For every piece of information you use, you MUST include a citation to the source document or chunk ID (e.g., [Source: Document A, Page 5] or [Chunk ID: 123]). This is non-negotiable for verifiability
        Context: {context_text}
        Question: {user_question}
        """
    else:
        prompt_template = f"""
        Answer the question. If you don't have enough information, just have a normal conversation.
        Question: {user_question}
        """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if docs_by_type["images"]:
        print(f"[DEBUG build_prompt] Adding {len(docs_by_type['images'])} images to prompt")
        for i, image in enumerate(docs_by_type["images"]):
            print(f"[DEBUG build_prompt] Image {i}: len={len(image)}")
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            })
    else:
        print("[DEBUG build_prompt] No images found in context")

    print("[DEBUG build_prompt] Complete =======================>>>>>")
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

RELEVANCE_THRESHOLD = 0.3

def get_vectorstore():
    embeddings = OllamaEmbeddingsCustom(model="nomic-embed-text")
    client = get_chroma_client()
    
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

def retrieve_with_scores(query: str):
    vectorstore = get_vectorstore()
    
    results = vectorstore.similarity_search_with_relevance_scores(query, k=1)
    
    print(f"\n[DEBUG retrieve] Query: {query}")
    print(f"[DEBUG retrieve] Found {len(results)} results")
    
    relevant_docs = []
    for doc, score in results:
        print(f"[DEBUG retrieve] Score: {score:.4f}, Doc: {doc.page_content[:50]}...")
        if score >= 0:
            doc_id = doc.metadata.get(ID_KEY)
            if doc_id:
                raw_data = store.mget([doc_id])
                if raw_data[0]:
                    relevant_docs.append(raw_data[0])
                    print(f"[DEBUG retrieve] Added doc with score {score:.4f}")
        else:
            print(f"[DEBUG retrieve] Skipped (below threshold {RELEVANCE_THRESHOLD})")
    
    print(f"[DEBUG retrieve] Returning {len(relevant_docs)} relevant docs\n")
    return relevant_docs

def get_rag_chain():
    chain_with_sources = {
        "context": RunnableLambda(retrieve_with_scores) | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            # | ChatOllama(model="llava", temperature=0)
            | ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0, api_key=os.getenv("GOOGLE_API_KEY", ""))
            | StrOutputParser()
        )
    )
    
    return chain_with_sources

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/clear")
def clear_collection():
    global store
    client = get_chroma_client()
    
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[DEBUG] Deleted collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"[DEBUG] No collection to delete: {e}")
    
    store = InMemoryStore()
    print("[DEBUG] Cleared InMemoryStore")
    
    return {"status": "ok", "message": f"Collection '{COLLECTION_NAME}' cleared"}

@app.post("/reindex")
def reindex_data():
    global store
    store = InMemoryStore()
    data, summaries = parse_pdf_and_get_data()
    result = store_vector(store, data, summaries, COLLECTION_NAME)
    return {"status": "ok", "result": result}

@app.post("/index")
def index_custom_data(request: IndexRequest):
    global store
    store = InMemoryStore()
    
    data = request.data.copy()
    summaries = request.summaries.copy()
    
    if IMAGE_PATHS:
        print(f"[DEBUG] Processing {len(request.images_paths)} image paths...")
        result_image = extract_and_chunk_images(IMAGE_PATHS, chunk_size=500, chunk_overlap=50)
        
        if "image" not in data:
            data["image"] = []
        if "image" not in summaries:
            summaries["image"] = []
            
        data["image"].extend(result_image["images"])
        summaries["image"].extend(result_image["summaries"])
        
        print(f"[DEBUG] Added {result_image['chunk_count']} chunks from images")
    
    result = store_vector(store, data, summaries, COLLECTION_NAME)
    return {"status": "ok", "result": result}

@app.get("/stats")
def get_stats():
    client = get_chroma_client()
    
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
    except Exception:
        count = 0
    
    store_keys = list(store.yield_keys())
    
    return {
        "collection": COLLECTION_NAME,
        "vectorstore_count": count,
        "inmemory_store_keys": len(store_keys),
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print(f"\n{'='*50}")
    print(f"[DEBUG chat] Received message: {request.message}")
    print(f"{'='*50}")
    
    chain = get_rag_chain()
    result = chain.invoke(request.message)
    
    print(f"\n[DEBUG chat] Response preview: {result['response'][:200]}...")
    print(f"[DEBUG chat] Context texts: {len(result['context']['texts'])}")
    print(f"[DEBUG chat] Context images: {len(result['context']['images'])}")
    
    images = [
        ContextImage(data_url=f"data:image/jpeg;base64,{img}")
        for img in result["context"]["images"][:3]
    ]
    
    return ChatResponse(
        response=result["response"],
        context_texts=result["context"]["texts"][:3],
        context_images=images,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)