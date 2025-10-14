# DocuMind 1.0

DocuMind 1.0 is a smart, easy-to-use document-based knowledge retrieval and question-answering system. It helps you search across PDFs and text files and get detailed, human-like answers using advanced Large Language Models (LLMs). This project was developed as a placement project inspired by a real requirement from Unthinkable Solutions.

Everything in DocuMind runs in-memory, meaning your documents and embeddings are never stored on disk. This ephemeral design keeps your data private, stateless, and easy to reset.

---

## Features

* **Upload Multiple Documents:** Quickly upload PDFs or text files through a simple web interface.
* **In-Memory Embeddings:** Converts your documents into vectors using Google Gemini embeddings for fast retrieval.
* **Smart Chunking:** Splits large documents into sentence-aware chunks to improve accuracy when retrieving answers.
* **RAG-Based Answers:** Uses Retrieval-Augmented Generation to combine document knowledge with general information for precise answers.
* **Human-Friendly Responses:** Generates natural, readable answers with clear formatting, including bold highlights, bullet points, and paragraphs.
* **Automatic Cleanup:** Uploaded files are removed after processing, and FAISS memory is cleared when the app shuts down.
* **Stateless and Secure:** No persistent storage, everything works in-memory for safe testing and experimentation.

---

## Tech Stack

* Backend: Python and Flask
* Embeddings: Google Gemini API (embedding-gecko-001)
* Vector Store: FAISS (in-memory)
* Frontend: HTML, CSS, JavaScript for file upload and query interface
* Environment Management: dotenv for handling API keys
* Document Parsing: PyPDF2 for PDFs, UTF-8 reading for TXT/MD files

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

3. Add your `.env` file with API keys (Gemini, Flask secret, etc.).

4. Run the app:

```bash
python app.py
```

5. Open your browser and visit `http://localhost:5000` to start uploading documents and asking questions.

---

This version reads more like a friendly guide while keeping all the technical details intact.

If you want, I can also **add a "How it Works" section with a simple flow diagram and example screenshots** to make it even more engaging for someone visiting your GitHub repo.

Do you want me to do that next?
