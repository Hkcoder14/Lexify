import streamlit as st
from PIL import Image
import os
from pathlib import Path  # NEW IMPORT ADDED
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_groq import ChatGroq

# Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# NEW FUNCTION ADDED TO HANDLE IMAGE PATHS
def get_image_path(image_name):
    """Get absolute path to image in app folder"""
    return os.path.join(Path(__file__).parent, image_name)

# App Config
st.set_page_config(
    page_title="Lexify - Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - UPDATED PATH
def local_css(file_name):
    css_path = get_image_path(file_name)  # UPDATED TO USE NEW PATH HANDLER
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # Now properly looks in app folder

# Sidebar
with st.sidebar:
    # UPDATED IMAGE PATH HANDLING
    try:
        logo_path = get_image_path("logo.jpg")
        st.image(logo_path, width=200)
    except FileNotFoundError:
        st.warning("Logo image not found at expected location")
    
    st.title("Lexify")
    st.subheader("Your AI Legal Assistant for Indian Law")
    
    # ... rest of sidebar content unchanged ...

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Main App
def main():
    # Header - UPDATED IMAGE PATH HANDLING
    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            logo_path = get_image_path("logo.jpg")
            st.image(logo_path, width=120)
        except FileNotFoundError:
            st.warning("Logo image not found")
    with col2:
        st.title("Lexify - Indian Legal Assistant")
        st.caption("Powered by AI analysis of Indian Constitution, IPC, and Labour Laws")
    
    st.markdown("---")
    
    # Initialize components
    if 'vectorstore' not in st.session_state:
        with st.spinner("Loading legal knowledge base..."):
            try:
                # Embedding + LLM setup
                embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                llm = ChatGroq(api_key=groq_api_key, model="llama3-70b-8192")
                
                # Load Chroma vector DB
                persist_directory = "VectoreStore/chroma"
                vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
                
                # Retriever
                retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 10})
                
                # Step 1: Query Reformulation Prompt (EXACTLY as in query_groq.py)
                query_expansion_prompt = PromptTemplate.from_template("""
You are a legal query reformulator for Indian law. Users will ask casual or informal questions.

Your task:
- Reformulate the question into a formal legal query.
- Use correct legal terminology relevant to Indian laws such as IPC, CrPC, Labour Law, Cyber Law, Consumer Protection Act, etc.
- Be specific and precise.
- Do NOT include phrases like "as per your question" or explanations—just output the formal legal query.

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

Q: My employer didn't pay me for 2 months!
→ What are the legal remedies available under the Payment of Wages Act, 1936, and Indian Labour Law for non-payment of salary?

Q: My neighbor built a wall on my land!
→ What legal actions can be taken under Indian property law in case of encroachment by a neighbor?

Q: I bought a phone and it stopped working in 3 days!
→ What rights does a consumer have under the Consumer Protection Act, 2019, for a defective electronic product?

Q: My friend's ex is threatening to leak private photos!
→ What legal protections are available under IPC and the IT Act for threats of non-consensual sharing of private images?

Q: Someone is using my PAN card to take loans!
→ What legal remedies are available under the Information Technology Act and IPC for identity theft and fraudulent financial activity?

Now reformulate:

Original question: {query}

Reformulated legal query:
""")
                
                reformulation_chain = query_expansion_prompt | llm
                
                # Step 2: Custom Lawyer Chain of Thought Prompt (EXACTLY as in query_groq.py)
                lawyer_style_prompt = PromptTemplate.from_template("""
You are a highly trained Indian legal expert. You are given the following legal documents which may contain general references to laws.

Your task is to infer, explain, and reason like a lawyer based on legal principles from the context — even if the exact situation is not directly stated.

Always format your answer as follows:
1. 📘 Relevant Laws
2. 🔍 Legal Reasoning
3. ⚖️ Potential Remedies or Actions
4. 📝 Conclusion

Use formal legal language. Base everything ONLY on the context provided. If it's absolutely not possible to answer, say exactly: "I don't know."

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
                
                st.session_state.vectorstore = vectordb
                st.session_state.retriever = retriever
                st.session_state.reformulation_chain = reformulation_chain
                st.session_state.qa_chain = qa_chain
                
            except Exception as e:
                st.error(f"Failed to initialize legal knowledge base: {str(e)}")
                st.stop()
    
    # Chat interface
    st.subheader("Ask Your Legal Question")
    user_query = st.text_area(
        "Type your question in plain English (e.g., 'Can my employer fire me without notice?')",
        height=100,
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        submit_btn = st.button("Get Legal Analysis", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear History", use_container_width=True)
    
    if clear_btn:
        st.session_state.history = []
        st.experimental_rerun()
    
    if submit_btn and user_query:
        with st.spinner("Analyzing your legal question..."):
            try:
                # Step 1: Reformulate user query
                response_obj = st.session_state.reformulation_chain.invoke({"query": user_query})
                reformulated_query = response_obj.content.strip()
                
                # Step 2: Run LLM QA
                response = st.session_state.qa_chain.invoke({"query": reformulated_query})
                answer = response.get("result", "I don't know.")
                
                # Add to history
                st.session_state.history.append({
                    "question": user_query,
                    "reformulated": reformulated_query,
                    "answer": answer
                })
                
            except Exception as e:
                st.error(f"Error processing your query: {str(e)}")
    
    # Display history
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Your Legal Query History")
        
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q: {item['question']}", expanded=(i==0)):
                st.markdown(f"""
                <div class="legal-card">
                    <div class="reformulated">
                        <strong>🔍 Legal Interpretation:</strong> {item['reformulated']}
                    </div>
                    <div class="answer">
                        <strong>⚖️ Legal Analysis:</strong>
                        {item['answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Example questions
def show_example_questions():
    st.markdown("---")
    st.subheader("Example Legal Questions")
    
    examples = [
        "Can my boss reduce my salary randomly?",
        "What happens if someone hits me?",
        "My landlord is harassing me, what can I do?",
        "Can I be fired for taking a sick leave?",
        "Someone is blackmailing me on Instagram!",
        "My employer didn't pay me for 2 months!"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, use_container_width=True):
                st.session_state.query_input = example

show_example_questions()
main()