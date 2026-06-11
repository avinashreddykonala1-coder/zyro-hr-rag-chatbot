import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Zyro Dynamics HR Help Desk")
st.caption("Ask any HR policy question. Powered by RAG + Groq.")

# ── Out-of-scope guardrail ───────────────────────────────────
HR_KEYWORDS = [
    "leave", "salary", "policy", "employee", "work from home", "wfh",
    "remote", "performance", "review", "appraisal", "travel", "expense",
    "reimbursement", "onboarding", "separation", "resignation", "termination",
    "notice", "conduct", "harassment", "posh", "probation", "it policy",
    "data", "device", "compensation", "benefits", "ctc", "grade", "increment",
    "maternity", "paternity", "sick", "casual", "earned", "holiday",
    "overtime", "shift", "attendance", "payroll", "insurance", "pf",
    "gratuity", "bonus", "promotion", "transfer", "grievance", "disciplinary",
    "zyro", "hr", "human resource", "joining", "offer", "contract"
]

OUT_OF_SCOPE_RESPONSE = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics policy documents."
)

def is_hr_related(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in HR_KEYWORDS)

# ── Build RAG pipeline (cached so it only runs once) ─────────
@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_rag():
    # IMPROVED: smaller chunks for more precise retrieval
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # IMPROVED: MMR retrieval for diverse + relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=512
    )

    # IMPROVED: strict grounded prompt
    prompt = ChatPromptTemplate.from_template("""
You are a precise HR policy assistant for Zyro Dynamics Pvt. Ltd.

Your rules:
1. Answer ONLY using the context provided below from the HR policy documents.
2. Be specific — include exact numbers, days, weeks, percentages, and policy names when they appear in context.
3. Do NOT add any information not present in the context.
4. Do NOT make assumptions or guess.
5. If the answer is not found in the context, respond with EXACTLY:
   "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents."

Context:
{context}

Question: {question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'HR Policy')}]\n{d.page_content}"
            for d in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, rag_chain

retriever, rag_chain = build_rag()

# ── Chat history ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ───────────────────────────────────────────────
if question := st.chat_input("Ask an HR question..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):

            # Guardrail check
            if not is_hr_related(question):
                answer = OUT_OF_SCOPE_RESPONSE
            else:
                retrieved = retriever.invoke(question)
                context_text = " ".join(d.page_content for d in retrieved)

                if len(context_text.strip()) < 100:
                    answer = OUT_OF_SCOPE_RESPONSE
                else:
                    answer = rag_chain.invoke(question)

                    # Show source documents
                    if retrieved:
                        sources = list({
                            d.metadata.get("source", "HR Policy").split("/")[-1]
                            for d in retrieved
                        })
                        answer += f"\n\n📄 *Sources: {', '.join(sources)}*"

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
