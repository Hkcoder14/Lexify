import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from Text_Extraction import extract_and_split_all_pdfs
from langchain_community.embeddings import OllamaEmbeddings

# Set paths
pdf_folder = "Documents"
persist_directory = "VectoreStore/chroma"  # Make sure casing matches

# Ensure output directory exists
os.makedirs(persist_directory, exist_ok=True)

print("🔄 Step 1: Extracting and splitting text from PDFs...")
try:
    split_texts = extract_and_split_all_pdfs(pdf_folder)
    if not split_texts:
        raise ValueError("No PDFs found or failed to extract text.")
except Exception as e:
    print(f"❌ Error extracting or splitting PDFs: {e}")
    exit(1)

documents, metadatas = [], []

print("🧩 Step 2: Preparing documents and metadata...")
for filename, chunks in split_texts.items():
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "source": filename,
            "chunk": i,
        })

print(f"📄 Total chunks: {len(documents)}")

if not documents:
    print("❌ No text chunks to embed. Exiting.")
    exit(1)

print("🤖 Step 3: Creating embeddings with HuggingFace...")
try:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")
    exit(1)

print("💾 Step 4: Storing embeddings in Chroma...")
try:
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"✅ Embeddings successfully stored in '{persist_directory}'")
except Exception as e:
    print(f"❌ Failed to store embeddings: {e}")
    exit(1)
