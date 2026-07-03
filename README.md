#  CourseMate-AI

CourseMate-AI is an AI-powered PDF Question Answering application that allows users to upload one or more PDF documents and ask natural language questions about their content. The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the uploaded documents and generate accurate answers using the Mistral Large Language Model.

Built with **Python, Streamlit, LangChain, Hugging Face Embeddings, ChromaDB, and Mistral AI**.

---

## 🚀 Features

* 📄 Upload one or multiple PDF documents
* 💬 Ask unlimited questions about uploaded PDFs
* 🤖 AI-powered answers using Mistral LLM
* 🔍 Semantic search with Hugging Face embeddings
* 📚 Retrieval-Augmented Generation (RAG) pipeline
* 🧠 Context-aware responses based only on uploaded documents
* 📑 Built-in PDF preview in the sidebar
* 💾 Chat history maintained during the session
* ⚡ Clean and interactive Streamlit interface

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* Mistral AI
* Hugging Face Embeddings
* ChromaDB
* PyPDFLoader
* Recursive Character Text Splitter
* python-dotenv

---

## 📂 Project Structure

```
CourseMate-AI/
│
├── .venv/
├── app.py
├── .env
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/prajnarao24/CourseMate-AI.git
cd CourseMate-AI
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Add your Mistral API key:

```
MISTRAL_API_KEY=your_api_key_here
```

### 5. Run the application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Upload one or more PDF documents.
2. The PDFs are loaded using **PyPDFLoader**.
3. Documents are split into smaller chunks using **RecursiveCharacterTextSplitter**.
4. Each chunk is converted into vector embeddings using **BAAI/bge-small-en-v1.5**.
5. Embeddings are stored in **ChromaDB**.
6. When a question is asked:

   * Relevant document chunks are retrieved using semantic search.
   * The retrieved context and user query are sent to the Mistral LLM.
   * The model generates an answer based only on the uploaded documents.

---
