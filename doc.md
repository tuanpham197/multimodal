# Chunking multi document type using multimodal (llm, vlm) for RAG pipeline

## 1. Extract partition document
 - khi nhận vào một tài liệu (pdf, docs, pptx,...) chúng ta cần có 1 bước xử lý tài liệu trước. Bằng việc bóc tách các thành phần của tài liệu như text, image, table,...
 - Sử dụng [unstructured] (https://docs.unstructured.io/open-source/core-functionality/partitioning) một thư viện của python để xử lý
 - Ví dụ dưới cho trường hợp file pdf:
 ```python

    from unstructured.partition.pdf import partition_pdf

    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables

        extract_image_block_types=["Image", "Table"],   # Add 'Table' to list to extract image of tables
        # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="by_title",          # or 'basic'
        max_characters=10000,                  # defaults to 500
        combine_text_under_n_chars=2000,       # defaults to 0
        new_after_n_chars=6000,

        # extract_images_in_pdf=True,          # deprecated
    )
 ```

## 2. Summary the image, table,.. using llm
## 3. Chunking summary
## 4. Store to vector database
