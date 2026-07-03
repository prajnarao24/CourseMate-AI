import os
import base64
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="CourseMate", layout="wide")
st.title("CourseMate")
st.write("Upload PDF(s) and ask unlimited questions about them.")

# ---------------- SESSION STATE ---------------- #
defaults = {
    "retriever": None,
    "llm": None,
    "prompt": None,
    "messages": [],
    "pdf_names": None,     # sorted list of filenames currently processed into the vectorstore
    "chat_active": True,
    "pdf_files": {},       # name -> raw bytes, kept purely for the sidebar preview
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- SIDEBAR: PREVIOUSLY UPLOADED PDFs ---------------- #
with st.sidebar:
    st.header("Uploaded PDFs")
    if st.session_state.pdf_files:
        selected_pdf = st.selectbox(
            "Preview a PDF",
            options=list(st.session_state.pdf_files.keys())
        )
        if selected_pdf:
            pdf_bytes = st.session_state.pdf_files[selected_pdf]
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.caption("No PDFs uploaded yet.")

# ---------------- PDF UPLOAD ---------------- #
uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    current_pdf_names = sorted([file.name for file in uploaded_files])

    # Keep bytes around for sidebar preview, regardless of whether we reprocess
    for file in uploaded_files:
        st.session_state.pdf_files[file.name] = file.getvalue()

    # Only re-run the (expensive) embedding pipeline if the set of PDFs changed
    if st.session_state.pdf_names != current_pdf_names:

        with st.spinner("Processing PDFs..."):

            temp_dir = tempfile.mkdtemp()
            all_docs = []

            for uploaded_file in uploaded_files:
                pdf_path = os.path.join(temp_dir, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                all_docs.extend(docs)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(all_docs)

            embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

            vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)

            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
            )

            st.session_state.llm = ChatMistralAI(model="mistral-small-2506")

            st.session_state.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant.

Use ONLY the provided context.

If the answer is not found in the context, say:

'I could not find the answer in the uploaded documents.'"""),
                ("human", """
Context:
{context}

Question:
{question}
""")
            ])

            st.session_state.pdf_names = current_pdf_names
            st.session_state.messages = []
            st.session_state.chat_active = True

        st.success(f"{len(uploaded_files)} PDF(s) processed successfully!")

    # ---------------- DISPLAY CHAT ---------------- #
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---------------- CHAT ---------------- #
    if st.session_state.chat_active:
        question = st.chat_input("Ask a question (type 0 to end chat)")

        if question:
            if question.strip() == "0":
                st.session_state.chat_active = False
                st.chat_message("assistant").markdown(
                    "Chat ended. Upload another PDF to start a new conversation."
                )
                st.stop()

            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            docs = st.session_state.retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)

            final_prompt = st.session_state.prompt.invoke({
                "context": context,
                "question": question
            })

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.llm.invoke(final_prompt)
                st.markdown(response.content)

            st.session_state.messages.append({"role": "assistant", "content": response.content})

    else:
        st.info("Conversation ended. Upload another PDF to begin again.")

else:
    st.info("Upload one or more PDFs to get started.")