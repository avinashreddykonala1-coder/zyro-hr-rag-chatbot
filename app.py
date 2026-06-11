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

st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="🤖"
)

st.title("🤖 Zyro Dynamics HR Help Desk")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


@st.cache_resource
def build_rag():

    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10
        }
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_template("""
You are an HR Help Desk assistant for Zyro Dynamics.

Answer ONLY from the provided context.

If the answer is not present in the context say:

I could not find this information in the Zyro Dynamics HR policy documents.

Context:
{context}

Question:
{question}

Answer:
""")

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

question = st.text_input(
    "Ask your HR question"
)

if question:

    answer = rag_chain.invoke(question)

    st.markdown("### Answer")
    st.write(answer)
