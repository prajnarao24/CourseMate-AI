
import os
import tempfile
import time

import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ---------------- ENVIRONMENT ---------------- #

load_dotenv()


# ---------------- PAGE CONFIGURATION ---------------- #

st.set_page_config(
    page_title="CourseMate",
    layout="wide"
)


# ---------------- GROQ API KEY ---------------- #

groq_api_key = os.getenv("GROQ_API_KEY")

# Read from Streamlit Cloud secrets if not found locally
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None


# ---------------- APPLICATION TITLE ---------------- #

st.title("CourseMate")
st.write("Upload PDF(s) and ask unlimited questions about them.")


# ---------------- SESSION STATE ---------------- #

defaults = {
    "retriever": None,
    "llm": None,
    "prompt": None,
    "messages": [],
    "pdf_names": None,
    "chat_active": True,
    "pdf_files": {},
    "last_api_call": 0.0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- PDF UPLOAD ---------------- #

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    current_pdf_names = sorted(
        [file.name for file in uploaded_files]
    )

    # Store uploaded PDF bytes
    for file in uploaded_files:
        st.session_state.pdf_files[file.name] = file.getvalue()

    # Process PDFs only when filenames change
    if st.session_state.pdf_names != current_pdf_names:

        try:
            with st.spinner("Processing PDFs..."):

                temp_dir = tempfile.mkdtemp()
                all_docs = []

                for uploaded_file in uploaded_files:

                    pdf_path = os.path.join(
                        temp_dir,
                        uploaded_file.name
                    )

                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()

                    all_docs.extend(docs)

                # Split documents
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(all_docs)

                if not chunks:
                    st.error(
                        "No readable text found in the uploaded PDFs."
                    )
                    st.stop()

                # Create embeddings
                embedding_model = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5"
                )

                # Create vector database
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model
                )

                # Create retriever
                st.session_state.retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 3,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )

                # Check Groq API key
                if not groq_api_key:
                    st.error(
                        "GROQ_API_KEY is missing. "
                        "Please configure it in Streamlit Cloud Secrets "
                        "or your local .env file."
                    )
                    st.stop()

                # Initialize Groq LLM
                st.session_state.llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    groq_api_key=groq_api_key,
                    temperature=0.2,
                    max_retries=0
                )

                # Create prompt
                st.session_state.prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """You are a helpful AI assistant.

Use ONLY the provided context.

If the answer is not found in the context, say:

'I could not find the answer in the uploaded documents.'"""
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
                ])

                st.session_state.pdf_names = current_pdf_names
                st.session_state.messages = []
                st.session_state.chat_active = True

            st.success(
                f"{len(uploaded_files)} PDF(s) processed successfully!"
            )

        except Exception as e:
            st.error(f"PDF processing error: {str(e)}")
            st.stop()


    # ---------------- DISPLAY CHAT HISTORY ---------------- #

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # ---------------- CHAT ---------------- #

    if st.session_state.chat_active:

        question = st.chat_input(
            "Ask a question (type 0 to end chat)"
        )

        if question:

            if question.strip() == "0":

                st.session_state.chat_active = False

                st.chat_message("assistant").markdown(
                    "Chat ended. Upload another PDF to start a new conversation."
                )

                st.stop()

            # Display user question
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            try:

                # Retrieve relevant document chunks
                docs = st.session_state.retriever.invoke(question)

                context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                # Prepare prompt
                final_prompt = st.session_state.prompt.invoke({
                    "context": context,
                    "question": question
                })

                # Generate answer
                with st.chat_message("assistant"):

                    with st.spinner("Thinking..."):

                        # Maintain spacing between requests
                        elapsed = (
                            time.time()
                            - st.session_state.last_api_call
                        )

                        if elapsed < 1.2:
                            time.sleep(1.2 - elapsed)

                        st.session_state.last_api_call = time.time()

                        response = st.session_state.llm.invoke(
                            final_prompt
                        )

                    answer = response.content

                    st.markdown(answer)

                # Save answer in chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(
                    f"An unexpected error occurred: {str(e)}"
                )

    else:

        st.info(
            "Conversation ended. Upload another PDF to begin again."
        )

else:

    st.info("Upload one or more PDFs to get started.")
