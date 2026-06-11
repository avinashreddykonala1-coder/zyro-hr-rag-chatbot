import streamlit as st

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🤖")

st.title("🤖 Zyro Dynamics HR Help Desk")

st.write(
    "Ask questions about leave policy, WFH policy, compensation, onboarding, travel policy, and other HR topics."
)

question = st.text_input("Enter your HR question:")

if question:
    st.write("Question:", question)
    st.info("RAG response will be connected here in the next step.")
