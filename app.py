import streamlit as st
import re
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
    "zyro", "acrux",
    "hr", "human resource", "joining", "offer", "contract",
    "payday", "pay day", "credited", "health", "medical", "pip",
    "annual", "eligible", "eligib"
]

OUT_OF_SCOPE_KEYWORDS = [
    "apply for a job", "recruitment", "hiring process",
    "product features", "acruxcrm", "salesforce", "compare it with",
    "revenue", "financially", "zoho", "freshworks"
]

OUT_OF_SCOPE_RESPONSE = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics policy documents."
)

REFUSAL_PHRASE = "I'm sorry, I can only answer HR-related questions"


def is_hr_related(question: str) -> bool:
    q = question.lower()
    if any(kw in q for kw in OUT_OF_SCOPE_KEYWORDS):
        return False
    return any(kw in q for kw in HR_KEYWORDS)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning model output"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


# ── Build RAG pipeline (cached so it only runs once) ─────────
@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_rag():
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 30,
            "lambda_mult": 0.5
        }
    )

    # NOTE: Update "model" below to match exactly what you used in Kaggle Cell 9
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model="qwen/qwen3-32b",   # 
        temperature=0,
        max_tokens=2048
    )

    prompt = ChatPromptTemplate.from_template("""
You are a precise HR policy assistant for Zyro Dynamics Pvt. Ltd.

IMPORTANT: "Acrux Dynamics" and "Zyro Dynamics" are the same company.

Your rules:
1. Answer ONLY using the context provided below.
2. Do NOT start with "According to..." or "As per the policy..."
3. Give COMPLETE answers — include ALL numbers, dates, tables, and details.
4. For notice period — show full grade-wise table (L1-L3, L4-L6, L7-L9, L10).
5. For WFH — list ALL types with eligibility criteria.
6. For leave questions — give EXACT numbers from the leave entitlement table.
7. Never summarize a table — reproduce it fully.
8. NEVER invent or hallucinate ANY information not present in the context.
9. If the question is about job applications, recruitment, product features, financials, or competitors — respond EXACTLY with:
   "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents."
10. If answer is not in context, respond EXACTLY with:
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
                    answer = strip_thinking(answer)  # remove <think> blocks

                    # Retry if LLM refused on a valid HR question
                    if REFUSAL_PHRASE in answer and is_hr_related(question):
                        retry_question = (
                            f"{question}\n\n"
                            "(Note: 'Acrux Dynamics' and 'Zyro Dynamics' are the same company. "
                            "Please look carefully through the entire context for the answer.)"
                        )
                        answer = rag_chain.invoke(retry_question)
                        answer = strip_thinking(answer)

                    # Show source documents
                    if retrieved:
                        sources = list({
                            d.metadata.get("source", "HR Policy").split("/")[-1]
                            for d in retrieved
                        })
                        answer += f"\n\n📄 *Sources: {', '.join(sources)}*"

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
