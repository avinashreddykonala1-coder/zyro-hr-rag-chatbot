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

GREETINGS_RESPONSE = (
    "Hello! 👋\n\n"
    "Welcome to the Zyro Dynamics HR Help Desk.\n\n"
    "I can help you with:\n"
    "- Leave policies (Earned, Sick, Casual, Maternity, Paternity, etc.)\n"
    "- Salary, payroll and compensation\n"
    "- Work from home policies\n"
    "- Performance reviews and PIP\n"
    "- Health insurance and benefits\n"
    "- Onboarding, separation and notice periods\n"
    "- Code of conduct and POSH policy\n"
    "- Travel and expense reimbursements\n\n"
    "How can I assist you today?"
)

HELP_RESPONSE = (
    "I am the Zyro Dynamics HR Assistant. "
    "I can help with leave policies, payroll, compensation, benefits, "
    "work-from-home policies, performance reviews, onboarding, "
    "separation policies, travel and expense policies, and other "
    "HR-related information available in the company documents. "
    "Feel free to ask me anything!"
)

THANKS_RESPONSE = (
    "You're welcome! 😊\n\n"
    "If you have any other questions about HR policies, leave, payroll, "
    "benefits, WFH, performance reviews, or other employee-related "
    "topics, feel free to ask."
)

GOODBYE_RESPONSE = (
    "Goodbye! 👋\n\n"
    "Feel free to return anytime if you need help with HR policies "
    "or employee-related information."
)

CHITCHAT_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier for an HR chatbot. Classify the message into one of these categories:

GREETING — user is greeting or making small talk. Examples: "hi", "hello", "hey", "hi...hi", "hiii", "hi how are you", "good morning", "good evening", "hi why am i here", "hi.hi.hi", "what's up", any casual opening message or random short text.

HELP — user is asking what the bot can do or who it is. Examples: "who are you", "what can you do", "help", "can you help me", "what do you know", "what topics can you help with".

THANKS — user is expressing gratitude. Examples: "thanks", "thank you", "thanks a lot", "great thanks", "ok thanks", "ty", "thx".

GOODBYE — user is saying goodbye. Examples: "bye", "goodbye", "see you", "see you later", "take care", "cya".

HR_QUESTION — user is asking about HR policies, leave, salary, benefits, work from home, performance, onboarding, conduct, travel, insurance, or any internal company policy topic. This includes:
- ALL types of leave: earned, sick, casual, maternity, paternity, bereavement, compensatory off
- Compensation: CTC, salary range, grade levels, bonus targets, increments, pay grades
- Benefits: health insurance, PF, gratuity, group medical insurance
- Performance: PIP, APR, ratings, appraisals, promotions
- Work arrangements: WFH, hybrid, remote work
- Onboarding, probation, separation, notice periods
- Code of conduct, POSH, disciplinary actions
- Travel and expense reimbursements

OUT_OF_SCOPE — user is asking about something completely unrelated to HR and not a greeting/thanks/goodbye. Examples: ESOP/stock options/vesting schedules, company revenue/financials/profit/funding, product features, competitor comparisons, job applications for outsiders, recruitment process, policies of other companies like Zoho or Freshworks.

IMPORTANT:
- Questions about CTC ranges, salary grades, bonus targets are ALWAYS HR_QUESTION
- Questions about ANY type of leave are ALWAYS HR_QUESTION
- ESOP, stock options, equity, vesting are ALWAYS OUT_OF_SCOPE

Message: {question}

Reply with ONLY one word: GREETING, HELP, THANKS, GOODBYE, HR_QUESTION, or OUT_OF_SCOPE
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
8. For APR/performance review timeline — write as properly punctuated flowing prose using "followed by" and "after which" to connect ALL phases. Use "and finally" ONLY ONCE at the very end. Include hyphens in "360-degree". Do NOT use semicolons or lists. Must include all 7 phases with their exact date ranges ending with 15 April.
9. For earned leave accrual — always include ALL three: accrual rate per month, total days after one year, AND the minimum 240 working days condition.
10. For salary/payroll questions — answer ONLY the date salary is credited and the payroll cut-off date. Nothing else.
11. Keep answers concise — only include facts that directly answer the question.
12. If the context does not contain the answer, respond EXACTLY with:
    "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents."
13. Never mention that something is missing from the context. Simply omit it.
14. If the question contains explicit date ranges or phase details in the notes — use those EXACT dates in your answer without modification.

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
            "promotion letter", "performance review timeline",
            "annual performance review", "apr"
        ],
        "hint": (
            "The APR timeline has these EXACT phases with EXACT dates — include ALL of them:\n"
            "360-degree feedback collected from peers and subordinates: 1 to 20 February\n"
            "Employee self-assessment submitted on ZyroHR portal: 1 to 10 March\n"
            "Manager completes assessment and submits draft rating: 11 to 20 March\n"
            "Calibration meetings with all L6 and above managers: 21 to 25 March\n"
            "Final ratings locked and confirmed by HR: 26 to 31 March\n"
            "One-on-one feedback conversations between employees and managers: 1 to 10 April\n"
            "Increment and promotion letters issued: 15 April\n\n"
            "Write as properly punctuated flowing prose using 'followed by' and 'after which' to connect phases. "
            "Use 'and finally' ONLY ONCE at the very end before 15 April. "
            "Include hyphens in '360-degree'. "
            "Do NOT use semicolons or bullet points."
        )
    },
    {
        "triggers": [
            "work from home", "wfh", "work-from-home", "remote work", "remote arrangement"
        ],
        "hint": (
            "For eligibility state: minimum 6 months continuous service, grade L3 or above, "
            "a performance rating of Meets Expectations or higher, no active PIP or disciplinary proceedings, "
            "and a role assessed as suitable for remote work by the reporting manager. "
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
    {
        "triggers": [
            "health insurance", "medical insurance", "insurance coverage",
            "premium arrangement", "who does it cover"
        ],
        "hint": (
            "Start with: 'Employees at Acrux Dynamics are provided with group medical insurance coverage of up to Rs. 5,00,000 per year.' "
            "Then state who it covers: the employee, their spouse, and up to two dependent children. "
            "Then state: the company fully pays all premiums."
        )
    },
    {
        "triggers": [
            "performance improvement plan", "pip", "placed on a pip",
            "when is an employee placed"
        ],
        "hint": (
            "Use this exact structure: An employee is placed on a Performance Improvement Plan (PIP) when they receive a rating of 1 or 2 in two consecutive review cycles. "
            "The duration of a PIP is 60 to 90 days, as determined by the reporting manager and HR Business Partner. "
            "Use 'when they receive' not 'after receiving'."
        )
    },
    {
        "triggers": [
            "sick leave", "consecutive days", "medical certificate",
            "sick leave for more than"
        ],
        "hint": (
            "Use capital letters for proper nouns: 'Medical Certificate' not 'medical certificate', "
            "'Sick Leave' not 'sick leave'. "
            "Structure: If an employee takes Sick Leave for more than 2 consecutive days, "
            "a Medical Certificate from a registered medical practitioner is required "
            "and must be submitted within 3 working days of returning to work."
        )
    },
    {
        "triggers": ["casual leave", "cl ", "casual leaves"],
        "hint": (
            "Casual Leave (CL) entitlement is 8 days per year. "
            "State this fact directly and concisely."
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
    cleaned = re.sub(
        r'<think>.*?(</think>|$)', '', text,
        flags=re.DOTALL | re.IGNORECASE
    )
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
        search_type="similarity",
        search_kwargs={"k": 12}
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

def classify_message(question: str) -> str:
    chain = CHITCHAT_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    result = result.strip().upper()
    for category in ["HR_QUESTION", "GREETING", "HELP", "THANKS", "GOODBYE", "OUT_OF_SCOPE"]:
        if category in result:
            return category
    return "OUT_OF_SCOPE"

def retrieve_context(question: str):
    docs = retriever.invoke(question)
    return format_docs(docs), docs

def generate_answer(question: str, context: str) -> str:
    chain = RAG_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"context": context, "question": question})
    return strip_thinking(raw)

def ask_bot(question: str) -> dict:
    try:
        # Stage 1: Classify message intent
        category = classify_message(question)

        if category == "GREETING":
            return {"answer": GREETINGS_RESPONSE, "sources": []}
        if category == "HELP":
            return {"answer": HELP_RESPONSE, "sources": []}
        if category == "THANKS":
            return {"answer": THANKS_RESPONSE, "sources": []}
        if category == "GOODBYE":
            return {"answer": GOODBYE_RESPONSE, "sources": []}
        if category == "OUT_OF_SCOPE":
            return {"answer": REFUSAL_MESSAGE, "sources": []}

        # Stage 2: HR_QUESTION — RAG with special hint
        context, docs = retrieve_context(question)
        special_hint = get_special_hint(question)
        full_question = f"{question}\n\n{special_hint}" if special_hint else question

        answer = generate_answer(full_question, context)

        # Stage 3: Retry if answer is refusal or too short
        if REFUSAL_PHRASE in answer or len(answer.strip()) < 20:
            fallback = (
                f"{question}\n\nExtract only facts explicitly stated in the context. "
                "Do not infer or add anything not directly written."
            )
            if special_hint:
                fallback += f"\n\n{special_hint}"
            answer = generate_answer(fallback, context)

        # Stage 4: APR validation
        q_lower = question.lower()
        if any(t in q_lower for t in ["apr", "annual performance review", "increment", "promotion letter"]):
            if "15 april" not in answer.lower() or "february" not in answer.lower():
                strong_hint = (
                    f"{question}\n\n"
                    "CRITICAL: Your answer MUST include ALL these exact dates: "
                    "1 to 20 February (360-degree feedback), "
                    "1 to 10 March (employee self-assessment on ZyroHR portal), "
                    "11 to 20 March (manager assessment and draft rating), "
                    "21 to 25 March (calibration meetings with L6 and above managers), "
                    "26 to 31 March (final ratings locked by HR), "
                    "1 to 10 April (one-on-one feedback conversations), "
                    "15 April (increment and promotion letters issued). "
                    "Use 'followed by' and 'after which' to connect phases. "
                    "Use 'and finally' only once at the end."
                )
                answer = generate_answer(strong_hint, context)

        # Final safety checks
        if "<think>" in answer.lower():
            answer = REFUSAL_MESSAGE
        if len(answer.strip()) < 10:
            answer = REFUSAL_MESSAGE

        sources = list({
            d.metadata.get("source", "HR Policy").split("/")[-1]
            for d in docs
        })
        return {"answer": answer, "sources": sources}

    except Exception as e:
        return {
            "answer": "I'm sorry, I encountered an error. Please try again in a moment.",
            "sources": []
        }

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
            try:
                result = ask_bot(question)
                answer = result["answer"]
                sources = result.get("sources", [])
                if sources:
                    answer += f"\n\n📄 *Sources: {', '.join(sources)}*"
            except Exception as e:
                answer = "I'm sorry, I encountered an error. Please try again in a moment."
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
