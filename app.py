import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="🤖"
)

st.title("🤖 Zyro Dynamics HR Help Desk")

st.write(
    "Ask questions about Leave Policy, WFH Policy, Compensation, Benefits, Travel, Onboarding and other HR topics."
)

# --------------------------------------------------
# API KEY
# --------------------------------------------------

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# --------------------------------------------------
# BUILD RAG
# --------------------------------------------------

@st.cache_resource
def build_rag():

    # Load PDFs
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10
        }
    )

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are an HR Help Desk assistant for Zyro Dynamics.

Answer ONLY using the provided context.

If the answer exists in the context, answer clearly and concisely.

If the answer does not exist in the context, respond exactly:

I could not find this information in the Zyro Dynamics HR policy documents.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = build_rag()

# --------------------------------------------------
# CHAT UI
# --------------------------------------------------

question = st.text_input(
    "Ask your HR question:"
)

if question:

    with st.spinner("Searching HR policies..."):

        try:
            answer = rag_chain.invoke(question)

            if not answer or len(answer.strip()) == 0:
                answer = (
                    "I could not find this information in the "
                    "Zyro Dynamics HR policy documents."
                )

            st.markdown("### Answer")
            st.write(answer)

        except Exception as e:
            st.error(str(e))
