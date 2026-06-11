import streamlit as st
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
    page_icon="🤖"
)

st.title("🤖 Zyro Dynamics HR Help Desk")

REFUSAL_MESSAGE = (
    "I could not find this information in the Zyro Dynamics HR policy documents."
)

@st.cache_resource
def build_rag():
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an HR assistant for Zyro Dynamics.
Answer ONLY using the provided context.
If the answer is not available in the context, respond exactly:
I could not find this information in the Zyro Dynamics HR policy documents.

Context: {context}

Question: {question}
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

    return retriever, rag_chain


retriever, rag_chain = build_rag()

question = st.text_input("Ask an HR question")

if question:
    answer = rag_chain.invoke(question)

    if not answer or not answer.strip():
        answer = REFUSAL_MESSAGE

    st.write(answer)
