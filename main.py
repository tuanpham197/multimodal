from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Image, CompositeElement
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Vector store
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

import chromadb
import os
import uuid


def summary_text(texts, tables):
    
    # Prompt
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOllama(
        model="llama3",
        temperature=0,
    )
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    # Summarize text
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
    return text_summaries, table_summaries

def summary_image(images):
    prompt_template = """Describe the image in detail. For context,
                  the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots."""
    
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | ChatOllama(model="llama3") | StrOutputParser()

    return chain.batch(images)
    
def filter_texts(chunks: list) -> list:
    return [
        chunk for chunk in chunks
        if isinstance(chunk, CompositeElement)
    ]

def filter_images_and_tables(chunks: list) -> tuple[list, list]:
    images, tables = [], []
    for chunk in chunks:
        if isinstance(chunk, Image):
            images.append(chunk)
        elif isinstance(chunk, Table):
            tables.append(chunk)
        elif isinstance(chunk, CompositeElement):
            for el in chunk.metadata.orig_elements or []:
                if isinstance(el, Image):
                    images.append(el)
                elif isinstance(el, Table):
                    tables.append(el)
    return images, tables

def filter_elements(chunks: list) -> dict:
    images, tables = filter_images_and_tables(chunks)
    return {
        "texts": filter_texts(chunks),
        "images": images,
        "tables": tables,
    }

def store_vertor(data, summaries):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST", "localhost"), port=int(os.getenv("CHROMA_PORT", "8001")))
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        client=client,
        collection_name="multi_modal_rag",
        embedding_function=embeddings,
    )

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in data['text']]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(summaries['text'])
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, data['text'])))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in data['table']]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(summaries['table'])
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, summaries['table'])))

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in data['image']]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(summaries['image'])
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, data['image'])))

    # check
    docs = retriever.invoke(
        "who are the authors of the paper?"
    )
    for doc in docs:
        print(str(doc) + "\n\n" + "-" * 80)

def verify_store_data():
    pass

def main():

    output_path = "./content/"
    file_path = output_path + 'attention.pdf'

    # Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables

        extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
        # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="by_title",          # or 'basic'
        max_characters=10000,                  # defaults to 500
        combine_text_under_n_chars=2000,       # defaults to 0
        new_after_n_chars=6000,

        # extract_images_in_pdf=True,          # deprecated
    )

    elements = filter_elements(chunks)
    
    print(f"Total chunks: {len(chunks)}")
    print(f"Texts: {len(elements['texts'])}")
    print(f"Images: {len(elements['images'])}")
    print(f"Tables: {len(elements['tables'])}")
    images = [img.metadata.image_base64 for img in elements['images']]
    # print(summary_text(elements['texts'], elements['tables']))
    t, tb = summary_text(elements['texts'],elements['tables'])
    sum_img = summary_image(images)
    print("==========>> START STORE DATABASE <<==========")
    store_vertor(
        {
            'text': elements['texts'],
            'table': elements['tables'],
            'image': images,
        },
        {
            'text': t,
        'table': tb,
        'image': sum_img,
        }
    )

    print(
        "==========>> END STORE DATABASE <<=========="
    )


if __name__ == "__main__":
    main()
