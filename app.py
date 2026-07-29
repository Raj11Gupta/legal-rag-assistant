"""
Streamlit application for the Legal RAG Assistant.

Features:
- Upload a legal PDF.
- Build a FAISS vector index.
- Retrieve relevant legal context.
- Generate answers using Google's Gemini model.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from google import genai

from rag_index_builder import build_index_from_pdf
from tools import retrieve_legal_context

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3.6-flash"
FAISS_DIR = "./rag_faiss_store"

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please configure your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ---------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg,#e3f2fd,#fce4ec);
        padding:25px;
        border-radius:15px;
        text-align:center;
    ">
        <h1>⚖️ Legal RAG Assistant</h1>
        <p>
            Ask questions about legal documents using
            <b>Retrieval-Augmented Generation (RAG)</b> powered by
            <b>Gemini + FAISS + Hugging Face Embeddings</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------------------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload a Legal PDF",
    type=["pdf"],
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    try:
        with st.spinner("Building FAISS index..."):
            build_index_from_pdf(
                pdf_path,
                persist_dir=FAISS_DIR,
                source_name=uploaded_file.name,
            )

        st.success("✅ Document indexed successfully!")

    except Exception as e:
        st.error(f"Failed to build index.\n\n{e}")
        st.stop()

    st.divider()

    # -----------------------------------------------------------------
    # Question Answering
    # -----------------------------------------------------------------

    question = st.text_input("💬 Ask a legal question")

    if question:

        try:
            with st.spinner("Searching document..."):

                context, sources = retrieve_legal_context(question)

                prompt = f"""
You are an expert legal assistant.

Answer ONLY using the legal context provided below.

If the answer is not available in the context,
respond with:
"I couldn't find this information in the uploaded document."

Keep your answer concise, accurate, and professional.

Legal Context:
{context}

Question:
{question}
"""

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                )

            st.subheader("🧠 Answer")
            st.success(response.text)

            st.subheader("📚 Source Documents")

            if sources:
                for source in sources:
                    st.write(f"• {source}")
            else:
                st.write("No source information available.")

            with st.expander("🔍 Retrieved Context"):
                st.write(context)

        except Exception as e:
            st.error(f"Error while generating response.\n\n{e}")

else:
    st.info("📄 Upload a legal PDF to begin.")