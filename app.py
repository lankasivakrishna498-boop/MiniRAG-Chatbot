
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

st.set_page_config(page_title="Mini RAG Chatbot", page_icon="🤖")
st.title("🤖 Mini RAG Chatbot")
st.write("Ask questions about your documents!")

@st.cache_resource
def setup_rag():
    file_path = Path(__file__).parent / "documents.txt"

    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="mini_rag_streamlit"
    )

    llm = ChatOllama(model="llama3.2:3b", temperature=0)

    return vectorstore, llm

try:
    with st.spinner("Loading documents and AI model..."):
        vectorstore, llm = setup_rag()

    st.success("RAG chatbot is ready!")

    question = st.text_input("Ask your question:")

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Searching documents and generating answer..."):
                results = vectorstore.similarity_search(question, k=3)
                context = "\n\n".join(
                    doc.page_content for doc in results
                )

                prompt = f"""
Answer using only the context below.
If the answer is not available in the context, say so.

Context:
{context}

Question: {question}
Answer:
"""
                response = llm.invoke(prompt)
                st.subheader("AI Response")
                st.write(response.content)
        else:
            st.warning("Please enter a question.")

except Exception as e:
    st.error("The chatbot could not start.")
    st.exception(e)