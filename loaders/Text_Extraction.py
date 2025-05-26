import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def extract_pdf_text(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)

def extract_and_split_all_pdfs(folder_path, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    all_docs = {}
    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue

        path = os.path.join(folder_path, filename)
        text = extract_pdf_text(path)
        chunks = splitter.split_text(text)

        all_docs[filename] = chunks

    return all_docs
