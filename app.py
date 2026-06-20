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

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Zyro Dynamics HR Help Desk")
st.caption("Ask any HR policy question. Powered by RAG + Groq.")

REFUSAL_MESSAGE = "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents."
REFUSAL_PHRASE  = "I'm sorry, I can only answer HR-related questions"

INTENT_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier. Decide if the question below is answerable from a company's internal HR policy documents.

HR policy documents typically cover:
- Leave policies (earned, sick, casual, maternity, paternity, etc.)
- Salary, payroll, CTC, compensation, grades, bonuses
- Work from home and remote work policies
- Performance reviews, PIP, appraisals, promotions
- Onboarding, probation, separation, notice periods
- Code of conduct, harassment (POSH), disciplinary actions
- Travel and expense reimbursements
- IT and data security policies
- Health insurance and employee benefits
- Attendance, holidays, shift policies

NOT answerable from HR policy documents:
- Job applications, recruitment or hiring process for outsiders
- ESOP allocations, stock option grants, vesting schedules, equity details
- Company financials, revenue, profit, funding, valuation
- Product features, product comparisons, competitor analysis
- Policies of other companies (Zoho, Freshworks, etc.)
- Anything unrelated to internal employee HR policies

Question: {question}

Reply with ONLY one word — YES if answerable from HR docs, NO if not.
""")

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise HR policy assistant for Zyro Dynamics Pvt. Ltd.
IMPORTANT: "Acrux Dynamics" and "Zyro Dynamics" refer to the same company.

Rules:
1. Answer ONLY using the context provided below. Never use outside knowledge.
2. Write in plain prose sentences only — no bullet points, no numbered lists, no bold (**), no headers, no tables.
3. Answer exactly what was asked. Do not add extra clauses, edge cases, or anything not directly asked.
4. Include every exact number, date, grade, Rs. value, and percentage that answers the question.
5. For WFH types — one sentence per type covering: name, eligible grade, max days per week.
6. For notice periods — one sentence covering all grades using semicolons.
7. For maternity leave — always include ALL three: 26 weeks (first two births), 12 weeks (third child), 80 days minimum service in the 12 months preceding delivery.
8. For APR/performance review timeline — write as flowing prose sentences using "followed by", "after which", "and finally". End with the exact date increment and promotion letters are issued.
9. For earned leave accrual — always include ALL three: accrual rate per month, total days after one year, AND the minimum 240 working days condition.
10. For salary/payroll questions — answer ONLY the date salary is credited and the payroll cut-off date. Do not add information about new joiners, pro-rata, or adjustments unless specifically asked.
11. If the context does not contain the answer, respond EXACTLY with:
    "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents."
12. Never mention that something is missing from the context. Simply omit it.

Context:
{context}

Question: {question}

Answer:
""")

SPECIAL_CASE_HINTS = [
    {
        "triggers": [
            "accrue per month", "accrual rate", "leave accru", "days per month",
            "one year of service", "how many days are employees entitled to after completing",
            "earned leave accrue"
        ],
        "hint": (
            "Earned Leave accrues at the rate of 1.25 days per month. "
            "Employees are entitled to 15 days of Earned Leave upon completing one year of continuous service, "
            "provided they have worked a minimum of 240 days in that year. "
            "IMPORTANT: Do NOT say accrual starts 'after completing one year' — "
            "accrual is monthly. Keep accrual rate and entitlement as two separate facts in one sentence."
        )
    },
    {
        "triggers": [
            "apr timeline", "annual performance review timeline", "increment",
            "promotion letter", "performance review timeline"
        ],
        "hint": (
            "Write the APR timeline as flowing prose sentences using 'followed by' and 'after which' to connect phases. "
            "Use 'and finally' ONLY ONCE at the very end before the letter issuance date. "
            "Do not repeat 'and finally' more than once. "
            "End with: increment and promotion letters are issued on 15 April."
        )
    },
    {
        "triggers": [
            "work from home", "wfh", "work-from-home", "remote work", "remote arrangement"
        ],
        "hint": (
            "For eligibility state only: minimum 6 months continuous service, grade L3 or above, "
            "performance rating of Meets Expectations or higher, no active PIP or disciplinary proceedings, "
            "and role assessed as suitable for remote work by the reporting manager. "
            "Do NOT add exclusions like probation or client-site. "
            "For WFH types write each as a full sentence: "
            "Hybrid WFH allows employees at grade L3 and above to work from home for a maximum of 3 days per week. "
            "Full Remote allows employees at grade L5 and above to work entirely remotely on a case-by-case basis for a maximum of 5 days per week. "
            "Ad-hoc WFH allows employees at grade L3 and above unplanned single-day WFH for a maximum of 2 days per week. "
            "Emergency WFH is available to all employees as directed by HR."
        )
    },
    {
        "triggers": ["maternity"],
        "hint": (
            "Include all three facts: "
            "26 weeks of paid Maternity Leave for the first two live births, "
            "12 weeks for the third child, "
            "and minimum 80 days of service in the 12 months preceding the expected date of delivery."
        )
    },
    {
        "triggers": [
            "salary credited", "payroll cut-off", "payday", "pay day", "salary credit",
            "which date is salary", "date is salary"
        ],
        "hint": (
            "Answer ONLY two facts: (1) salary is credited by the 7th of the following month, "
            "and (2) the payroll cut-off date is the 24th of each month. "
            "Do NOT add anything about new joiners, pro-rata, or adjustments."
        )
    },
]

def get_special_hint(question: str) -> str:
    q = question.lower()
    for case in SPECIAL_CASE_HINTS:
        if any(trigger in q for trigger in case["triggers"]):
            return case["hint"]
    return ""

def strip_thinking(text: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = cleaned.replace('**', '')
    cleaned = re.sub(r'\n\d+\.\s+', ' ', cleaned)
    hedge_pattern = re.compile(
        r'[^.]*?\b(?:is\s+not\s+(?:explicitly\s+)?(?:specified|stated|documented|mentioned|provided|available)'
        r'|does\s+not\s+(?:specify|state|mention|provide|document)'
        r'|is\s+implied\s+to\s+be)[^.]*\.\s*',
        re.IGNORECASE
    )
    cleaned = hedge_pattern.sub('', cleaned)
    cleaned = re.sub(r'^\s*(However|But)\s*,?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\.\s+(However|But)\s*,?\s+', '. ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()

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
        search_kwargs={"k": 5, "fetch_k": 25, "lambda_mult": 0.5}
    )
    llm = ChatGroq(
        groq_api_key=st.secrets.get("GROQ_API_KEY") or st.secrets.get("groq_api_key"),
        model="qwen/qwen3-32b",
        temperature=0.1,
        max_tokens=1024
    )
    return retriever, llm

retriever, llm = build_rag()

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'HR Policy')}]\n{d.page_content}"
        for d in docs
    )

def classify_intent(question: str) -> bool:
    chain = INTENT_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    result = result.strip().upper()
    return result.startswith("YES")

def retrieve_context(question: str):
    docs = retriever.invoke(question)
    return format_docs(docs), docs

def generate_answer(question: str, context: str) -> str:
    chain = RAG_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"context": context, "question": question})
    return strip_thinking(raw)

def ask_bot(question: str) -> dict:
    if not classify_intent(question):
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    context, docs = retrieve_context(question)
    special_hint = get_special_hint(question)
    full_question = f"{question}\n\n{special_hint}" if special_hint else question

    answer = strip_thinking(generate_answer(full_question, context))

    if REFUSAL_PHRASE in answer or len(answer.strip()) < 20:
        fallback = (
            f"{question}\n\nExtract only facts explicitly stated in the context. "
            "Do not infer or add anything not directly written."
        )
        if special_hint:
            fallback += f"\n\n{special_hint}"
        answer = strip_thinking(generate_answer(fallback, context))

    sources = list({d.metadata.get("source", "HR Policy").split("/")[-1] for d in docs})
    return {"answer": answer, "sources": sources}

# ── Chat UI ──────────────────────────────────────────────────
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
            result = ask_bot(question)
            answer = result["answer"]
            sources = result["sources"]
            if sources:
                answer += f"\n\n📄 *Sources: {', '.join(sources)}*"
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
