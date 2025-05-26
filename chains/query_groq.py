import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_groq import ChatGroq

# Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Paths
persist_directory = "VectoreStore/chroma"

# Embedding + LLM setup
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(api_key=groq_api_key, model="llama3-70b-8192")

# Load Chroma vector DB
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Retriever
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 10})

# Step 1: Query Reformulation Prompt
query_expansion_prompt = PromptTemplate.from_template("""
You are a legal query reformulator for Indian law. Users will ask casual or informal questions.

Your task:
- Reformulate the question into a formal legal query.
- Use correct legal terminology relevant to Indian laws such as IPC, CrPC, Labour Law, Cyber Law, Consumer Protection Act, etc.
- Be specific and precise.
- Do NOT include phrases like “as per your question” or explanations—just output the formal legal query.

Examples:
Q: Can my boss reduce my salary randomly?
→ What are the legal provisions under Indian Labour Law regarding unauthorized salary deductions by employers?

Q: What happens if someone hits me?
→ What are the legal consequences under IPC for physical assault in India?

Q: How much do you get paid in India by law?
→ What is the minimum wage rate fixed under the Minimum Wages Act, 1948, in India?

Q: My landlord is harassing me, what can I do?
→ What legal remedies are available under Indian Rent Control Acts and IPC for landlord harassment?

Q: Police took my bike without any reason!
→ What are the legal rights of a vehicle owner under Indian law if the police seize a vehicle without a warrant?

Q: Can I be fired for taking a sick leave?
→ What protections are provided under Indian Labour Law against termination due to medical leave?

Q: My husband beats me, can I file a case?
→ What legal action can a wife take under Section 498A of IPC and the Domestic Violence Act in case of physical abuse by husband?

Q: Someone is blackmailing me on Instagram!
→ What legal remedies are available under the Information Technology Act, 2000, and IPC for online blackmail and cyber harassment?

Q: My internet provider isn't fixing my connection for days!
→ What legal recourse does a consumer have under the Consumer Protection Act, 2019, for deficiency in internet services?

Q: I want to divorce my wife, what are the steps?
→ What is the legal procedure for obtaining a divorce under the Hindu Marriage Act, 1955?

Q: A guy jumped a red light and hit my car!
→ What are the legal penalties under the Motor Vehicles Act, 1988, for violating traffic signals resulting in an accident?

Q: My employer didn’t pay me for 2 months!
→ What are the legal remedies available under the Payment of Wages Act, 1936, and Indian Labour Law for non-payment of salary?

Q: My neighbor built a wall on my land!
→ What legal actions can be taken under Indian property law in case of encroachment by a neighbor?

Q: I bought a phone and it stopped working in 3 days!
→ What rights does a consumer have under the Consumer Protection Act, 2019, for a defective electronic product?

Q: My friend’s ex is threatening to leak private photos!
→ What legal protections are available under IPC and the IT Act for threats of non-consensual sharing of private images?

Q: Someone is using my PAN card to take loans!
→ What legal remedies are available under the Information Technology Act and IPC for identity theft and fraudulent financial activity?

Now reformulate:

Original question: {query}

Reformulated legal query:
""")

reformulation_chain = query_expansion_prompt | llm

# Step 2: Custom Lawyer Chain of Thought Prompt
lawyer_style_prompt = PromptTemplate.from_template("""
You are a highly trained Indian legal expert. You are given the following legal documents which may contain general references to laws.

Your task is to infer, explain, and reason like a lawyer based on legal principles from the context — even if the exact situation is not directly stated.

Always format your answer as follows:
1. 📘 Relevant Laws
2. 🔍 Legal Reasoning
3. ⚖️ Potential Remedies or Actions
4. 📝 Conclusion

Use formal legal language. Base everything ONLY on the context provided. If it’s absolutely not possible to answer, say exactly: "I don't know."

📄 CONTEXT:
{context}

❓ QUESTION:
{question}
""")

# Create RetrievalQA with the updated legal reasoning prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": lawyer_style_prompt},
    return_source_documents=True
)

# Debugging tool to view top documents retrieved
def test_retrieval(query):
    print("\n🔍 Retrieved Documents Preview:")
    docs = retriever.get_relevant_documents(query)
    if not docs:
        print("⚠️ No relevant documents found.")
        return
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n📄 Document {i} Preview:\n{doc.page_content[:400]}...\n[Source: {doc.metadata.get('source')}]\n")

# Interactive chat
print("🧠 Ask me anything about Indian law (type 'exit' to quit):\n")

while True:
    user_query = input("You: ")
    if user_query.lower() == 'exit':
        break

    # Step 1: Reformulate user query
    response_obj = reformulation_chain.invoke({"query": user_query})
    reformulated_query = response_obj.content.strip()
    print(f"\n🔁 Reformulated Query:\n{reformulated_query}")

    # Debug: See top matching legal documents
    test_retrieval(reformulated_query)

    # Step 2: Run LLM QA
    try:
        response = qa_chain.invoke({"query": reformulated_query})
        answer = response.get("result", "I don't know.")
        print(f"\n📜 Answer:\n{answer}\n")
    except Exception as e:
        print(f"⚠️ Error during processing: {e}")
