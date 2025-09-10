import streamlit as st
from transformers import pipeline

st.title("📄 Document question answering")

st.write(
    "Upload a document below and ask a question about it. "
)

uploaded_file = st.file_uploader("Upload a document (.txt)", type=["txt"])

question = st.text_area(
    "Now ask a question about the document!",
    placeholder="What is this document about?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    document = uploaded_file.read().decode()

    # Load QA pipeline (this downloads model on first run)
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # Run the QA model
    result = qa_pipeline(question=question, context=document)

    st.write("**Answer:**")
    st.write(result['answer'])
