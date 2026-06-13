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

# Page configuration
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Zyro Dynamics HR Help Desk")
st.caption("Ask any HR policy question. Powered by RAG + Groq.")

# ---------------------------------------------------------------------------
# Guardrail keyword lists
# ---------------------------------------------------------------------------
HR_KEYWORDS = [
    "leave", "salary", "policy", "employee", "work from home", "wfh",
    "remote", "performance", "review", "appraisal", "travel", "expense",
    "reimbursement", "onboarding", "separation", "resignation", "termination",
    "notice", "conduct", "harassment", "posh", "probation", "it policy",
    "data", "device", "compensation", "benefits", "ctc", "grade", "increment",
    "maternity", "paternity", "sick", "casual", "earned", "holiday",
    "overtime", "shift", "attendance", "payroll", "insurance", "pf",
    "gratuity", "bonus", "promotion", "transfer", "grievance", "disciplinary",
    "zyro", "acrux", "hr", "human resource", "joining", "joiner", "offer", "contract",
    "payday", "pay day", "credited", "health", "medical", "pip",
    "annual", "eligible", "eligib",
    # ESOP / equity additions
    "esop", "stock option", "stock options", "vesting", "vest",
    "equity", "shares", "cliff"
]

OUT_OF_SCOPE_KEYWORDS = [
    "apply for a job", "recruitment", "hiring process",
    "product features", "acruxcrm", "salesforce", "compare it with",
    "revenue", "financially", "zoho", "freshworks"
]

OUT_OF_SCOPE_RESPONSE = "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents."
REFUSAL_PHRASE = "I'm sorry, I can only answer HR-related questions"


def is_hr_related(question: str) -> bool:
    q = question.lower()
    if any(kw in q for kw in OUT_OF_SCOPE_KEYWORDS):
        return False
    return any(kw in q for kw in HR_KEYWORDS)


def strip_thinking(text: str) -> str:
    """Removes model reasoning blocks, illegal markdown markers, numbered-list artifacts, and hedging sentences about missing info."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = cleaned.replace('**', '')
    cleaned = re.sub(r'\n\d+\.\s+', ' ', cleaned)

    # Remove standalone sentences that only state something is missing/undocumented
    hedge_pattern = (
        r'\b(?:The\s+)?(?:context|policy|document|provided\s+(?:context|information|materials))'
        r'[^.]*?\b(?:does\s+not\s+(?:specify|state|mention|provide|document)'
        r'|is\s+not\s+(?:explicitly\s+)?(?:documented|specified|stated|mentioned|provided|available)|is\s+implied\s+to\s+be)'
        r'[^.]*\.\s*'
    )
    cleaned = re.sub(hedge_pattern, '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# RAG pipeline (cached so it only builds once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_rag():
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.7}
    )

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=2048,
        reasoning_effort="none"
    )

    prompt = ChatPromptTemplate.from_template("""
You are a precise HR policy assistant for Zyro Dynamics Pvt. Ltd.

IMPORTANT: "Acrux Dynamics" and "Zyro Dynamics" are the same company.

Your rules:
1. Answer ONLY using the context provided below.
2. Write completely in plain prose sentences and paragraphs. Do NOT use markdown bold (**), markdown headers, HTML, tables, markdown lists, numbered lists (such as 1., 2., 3.), or bullet points under any circumstances.
3. Answer ONLY what is directly asked — do not add extra related information, tips, meta-notes, or "additional notes" beyond the question's scope.
4. Include ALL exact numbers, dates, calendar days, grade levels, currency values (Rs.), and percentages relevant to the question, written naturally within sentences.
5. For notice period questions — describe the grade-wise periods in sentence form (e.g., "L1 to L3 employees have a 30-day notice period, L4 to L6 have 60 days...").
6. For WFH or multi-type questions — describe each type in continuous sentence prose, separating items with commas or semicolons, not lists or new lines.
7. NEVER invent, assume, extrapolate, or hallucinate information not present in the context.
8. If the question is about job applications, recruitment, product features, financials, or competitors — respond EXACTLY with:
   "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents."
9. If the answer is not in context, respond EXACTLY with:
   "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents."
10. Never comment on what the context, policy, or document does or does not state, specify, document, or mention, and never write phrases like "is not explicitly stated", "is implied to be", "the context does not specify", or "is not documented". Do not mention missing information at all — simply omit it. For example, instead of writing "The duration is not specified, but it can be extended by 30 additional days", write "It can be extended by up to 30 additional days." State only the facts that ARE present, as confident, direct sentences.

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

# ---------------------------------------------------------------------------
# Bot wrapper with guardrails + retry-on-refusal logic
# ---------------------------------------------------------------------------
def ask_bot(question: str) -> str:
    if not is_hr_related(question):
        return OUT_OF_SCOPE_RESPONSE

    retrieved = retriever.invoke(question)
    context_text = " ".join(d.page_content for d in retrieved)

    if len(context_text.strip()) < 100:
        return OUT_OF_SCOPE_RESPONSE

    answer = rag_chain.invoke(question)
    answer = strip_thinking(answer)

    if REFUSAL_PHRASE in answer and is_hr_related(question):
        hint_question = (
            f"{question}\n\n"
            "(Note: Evaluate the context text comprehensively. Pull all numbers, dates, ranges, and explicit values directly.)"
        )
        answer = rag_chain.invoke(hint_question)
        answer = strip_thinking(answer)

    if retrieved:
        sources = list({d.metadata.get("source", "HR Policy").split("/")[-1] for d in retrieved})
        answer += f"\n\n📄 *Sources: {', '.join(sources)}*"

    return answer


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            answer = ask_bot(question)

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
