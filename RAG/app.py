import os
import tempfile
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        # Embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector DB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        st.success(
            f"PDF loaded successfully! ({len(chunks)} chunks created)"
        )

    # Initialize LLM
    llm = ChatMistralAI(
        model="mistral-small-latest"
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """
Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    question = st.text_input(
        "Ask a question about the PDF:"
    )

    if question:

        with st.spinner("Searching document..."):

            docs = retriever.invoke(question)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt_template.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved Context"):

            for i, doc in enumerate(docs):
                st.write(f"### Chunk {i+1}")
                st.write(doc.page_content[:1000])
