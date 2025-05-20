from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from groq import Groq
import os
from dotenv import load_dotenv
from langchain.llms.base import LLM
from pydantic import BaseModel, Field
from typing import Optional, List, Mapping, Any

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

class GroqLLM(LLM, BaseModel):
    api_key: str
    model_name: str
    client: Groq = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.client = Groq(api_key=self.api_key)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 512,
        }
        if stop:
            kwargs["stop_sequences"] = stop

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "groq"


# 1. Load Chroma Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="vectorstore/chroma",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Setup Groq LLM
llm = GroqLLM(
    api_key=api_key,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# 3. Prompt Template
prompt = PromptTemplate.from_template("""
Use the following legal context to answer the question as accurately as possible.
If you don't know the answer, just say you don't know.

Context: {context}
Question: {question}
""")

# 4. Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 5. Ask questions loop
while True:
    query = input("Ask a legal question: ")
    if query.lower() in ["exit", "quit"]:
        break

    # use invoke() to avoid deprecation warnings
    result = qa_chain.invoke({"query": query})
    print(f"\n📜 Answer:\n{result}\n")
