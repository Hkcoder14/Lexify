import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Paths
persist_directory = "VectoreStore/chroma"

# Embedding and LLM
embedding = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
llm = ChatGroq(api_key=groq_api_key, model="llama3-70b-8192")

# Load Vector Store
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Better Retriever: Using MMR and fetching top 10 results
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 10})

# Step 1: Query Reformulation Chain
query_expansion_prompt = PromptTemplate.from_template("""
You are a legal query reformulator. A user may ask a question in casual language.
Your job is to reformulate it into a formal legal search query using appropriate terminology,
relevant to Indian law (IPC, CrPC, Labor law, etc.).

Original question: {query}

Reformulated legal query:
""")
reformulation_chain = LLMChain(llm=llm, prompt=query_expansion_prompt)

# Step 2: QA Chain with custom prompt
custom_prompt = PromptTemplate.from_template("""
You are a legal assistant. Use only the provided context to answer.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# Debug tool: Show top retrieved documents
def test_retrieval(query):
    print("\n🔍 Retrieved Documents Preview:")
    docs = retriever.invoke(query)
    if not docs:
        print("⚠️ No relevant documents found.")
    for i, doc in enumerate(docs[:3], 1):  # Show top 3
        print(f"\n📄 Document {i} Preview:\n{doc.page_content[:400]}...\n[Source: {doc.metadata.get('source')}]\n")

# Main chat loop
print("🧠 Ask me anything about Indian law (type 'exit' to quit):\n")
while True:
    user_query = input("You: ")
    if user_query.lower() == 'exit':
        break

    # Reformulate query
    reformulated_query = reformulation_chain.run({"query": user_query})
    print(f"\n🔁 Reformulated Query:\n{reformulated_query}")

    # Debug: Show retrieved documents
    test_retrieval(reformulated_query)

    # Run QA chain
    result = qa_chain.run(reformulated_query)
    print(f"\n📜 Answer:\n{result}\n")
