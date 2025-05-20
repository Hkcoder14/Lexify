import fitz
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text+= page.get_text()
    doc.close()
    return text

def extract_and_split_all_pdfs(folder_path, chunk_size = 1000, chunk_overlap = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            full_text = extract_pdf(pdf_path)

            # Split into chunks
            chunks = splitter.split_text(full_text)
            split_docs[filename] = chunks

    return split_docs

if __name__ == "__main__":
    folder = "Documents"
    split_texts = extract_and_split_all_pdfs(folder)




