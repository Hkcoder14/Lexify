from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# Load the split chunks function from your module
from Text_Extraction import extract_and_split_all_pdfs

pdf_folder = "Documents"
persist_directory = "VectoreStore/chroma"

# Load chunks from PDFs
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

# Create Chroma vector store with persistence enabled
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory=persist_directory
)

print("✅ Embeddings stored in Chroma DB.")
