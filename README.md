# DocuMentor

DocuMentor is an AI-powered document chat assistant. It enables users to upload documents (PDF, DOCX, XLSX, TXT), process them into vector stores, and interactively ask questions about their content using a local LLM (Ollama with Mistral) and LangChain's retrieval-augmented generation (RAG) pipeline.

---

## Features

- **Multi-format Document Support:** Upload and process PDF, DOCX, XLSX, and TXT files.
- **Session-based Chat:** Each upload creates a new session with persistent chat history.
- **Semantic Search:** Uses vector embeddings for accurate document retrieval.
- **Local LLM Integration:** Runs queries using Ollama's Mistral model for privacy and speed.
- **Web Interface:** Simple Flask-based UI for uploading, chatting, and session management.
- **Extensible:** Modular codebase for easy extension to new document types or LLMs.

---

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running with the `mistral` model or any other model. You just need to change the code according to the model.
- All Python dependencies in `requirements.txt`

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd documentor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama with the Mistral model:**
   ```bash
   ollama run mistral
   ```

4. **Run the Flask app:**
   ```bash
   python app.py
   ```

5. **Open your browser:**  
   Go to [http://localhost:5000](http://localhost:5000)

---

## Usage

1. **Upload Documents:**  
   On the home page, upload one or more documents to start a new chat session.

2. **Chat:**  
   Ask questions about the uploaded documents. The assistant will answer using only the content from your files.

3. **Session Management:**  
   Previous sessions are listed on the home page. Click any session to revisit its chat and documents.

---

## Architecture

- **Flask:** Serves the web interface and API endpoints.
- **Session Manager:** Handles session creation, document storage, and chat history.
- **Ingest Pipeline:** Extracts and chunks text from uploaded documents, builds a FAISS vector store.
- **Agent:** Uses LangChain and Ollama to answer questions by retrieving relevant document chunks.
- **Vector Store:** Each session has its own FAISS vector store for isolation and speed.

**File Overview:**
- `app.py` — Main Flask app and routes
- `agent.py` — LangChain agent and RAG logic
- `ingest.py` — Document parsing and vectorization
- `session_manager.py` — Session and chat history management
- `vector_store.py` — Vector store loading utilities
- `templates/` — HTML templates
- `static/` — CSS and static assets
- `sessions/` — Per-session data storage

---

## Troubleshooting

- **Ollama not running:**  
  Ensure `ollama run mistral` is active before starting the app.

- **Dependency issues:**  
  Double-check Python version and run `pip install -r requirements.txt` again.

- **Large files:**  
  Processing very large documents may take time and memory.

---

## License

MIT License. See `LICENSE` file for details.
