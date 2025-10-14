# DocuMind 1.0

DocuMind 1.0 is a smart, easy-to-use document-based knowledge retrieval and question-answering system. It helps you search across PDFs and text files and get detailed, human-like answers using advanced Large Language Models (LLMs). This project was developed as a placement project inspired by a real requirement from **Unthinkable Solutions**.

Everything in DocuMind runs in-memory, meaning your documents and embeddings are never stored on disk. This ephemeral design keeps your data private, stateless, and easy to reset.

---

## Features

* Upload Multiple Documents: Quickly upload PDFs or text files through a simple web interface.
* In-Memory Embeddings: Converts your documents into vectors using **Google Gemini embeddings** for fast retrieval.
* Smart Chunking: Splits large documents into sentence-aware chunks to improve accuracy when retrieving answers.
* RAG-Based Answers: Uses Retrieval-Augmented Generation to combine document knowledge with general information for precise answers.
* Human-Friendly Responses: Generates natural, readable answers with bold highlights, bullet points, and paragraphs.
* Automatic Cleanup: Uploaded files are removed after processing, and FAISS memory is cleared on shutdown.
* Stateless and Secure: No persistent storage; everything works in-memory for safe testing and experimentation.

---

## Tech Stack

* Backend: Python and Flask
* Embeddings: Google Gemini API (`embedding-gecko-001`)
* Vector Store: FAISS (in-memory)
* Frontend: HTML, CSS, JavaScript for file upload and query interface
* Environment Management: dotenv for API keys
* Document Parsing: PyPDF2 for PDFs, UTF-8 reading for TXT/MD files

---

## How It Works

1. **Upload Documents:** Users upload PDFs or TXT/MD files through the web interface.
2. **Text Extraction & Chunking:** Files are parsed and split into smaller, sentence-aware chunks.
3. **Embedding Generation:** Each chunk is converted into a vector using Google Gemini embeddings.
4. **In-Memory Storage:** Vectors and metadata are stored in FAISS for fast retrieval.
5. **Querying:** Users ask a question, which is converted to an embedding and matched with document chunks.
6. **Answer Synthesis:** The LLM generates a human-friendly, detailed answer using the retrieved chunks and general knowledge.
7. **Clean-Up:** Uploaded files are deleted, and memory is cleared after use.

**Flow Diagram (Example):**

```
[User Upload] --> [Text Extraction] --> [Chunking] --> [Embedding (Gemini)] --> [FAISS Storage]
                                \
                                 --> [Query] --> [Retrieve Chunks] --> [LLM Answer] --> [User]
```

**Screenshots (Example Placeholder):**

* Upload interface screenshot: `static/screenshots/upload.png`
* Query interface screenshot: `static/screenshots/query.png`
* Sample answer screenshot: `static/screenshots/answer.png`

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/theindianarmytan05-create/DocuMind_1.0.git
cd DocuMind
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your `.env` file with your API keys:

```
GEMINI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=your_flask_secret
EMBEDDING_DIM=1536
```

4. Run the app:

```bash
python app.py
```

5. Open your browser and visit `http://localhost:5000` to start uploading documents and asking questions.

---

## Contributing

Feel free to submit issues, suggestions, or pull requests. All contributions are welcome.

---
