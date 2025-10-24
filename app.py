import streamlit as st
from edugen.rag import upsert_pdf
from edugen.core import generate_outline, generate_quiz

st.set_page_config(page_title="EduGen – AI Lesson & Quiz Builder", layout="wide")
st.title("📚 EduGen – AI Lesson & Quiz Builder")

with st.sidebar:
    st.header("Knowledge Base")
    up = st.file_uploader("Upload PDF(s) to ground lessons", type=["pdf"], accept_multiple_files=True)
    if st.button("Build Index") and up:
        total = 0
        for f in up:
            path = f".cache_{f.name}"
            with open(path, "wb") as out: out.write(f.read())
            total += upsert_pdf(path)
        st.success(f"Indexed {total} chunks.")

topic = st.text_input("Topic")
col1,col2 = st.columns(2)
with col1: gen_outline = st.checkbox("Generate Outline", True)
with col2: gen_quiz = st.checkbox("Generate Quiz", True)
go = st.button("Generate")

if go and topic.strip():
    if gen_outline:
        st.subheader("📘 Lesson Outline")
        st.write(generate_outline(topic))
    if gen_quiz:
        st.subheader("📝 Quiz")
        st.write(generate_quiz(topic))
else:
    st.info("Upload PDFs (optional), enter a topic, then Generate.")
