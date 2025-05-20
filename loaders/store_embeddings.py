from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os 
import pickle

#Load the split chunks
from Text_Extraction import extract_and_split_all_pdfs
pdf_folder = "Documents"
persist_directory = "VectoreStore/chroma"

#load chunks
split_texts = extract_and_split_all_pdfs(pdf_folder)

# Flatten the chunks (with metadata)
documents = []
metadatas = []

for filename, chunks in split_texts.items():
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({"source": filename, "chunk": i})

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vectorstore
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory=persist_directory
)

# Save to disk
vectorstore.persist()
print("✅ Embeddings stored in Chroma DB.")