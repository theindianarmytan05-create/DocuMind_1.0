# DocuMind 1.0

**DocuMind 1.0** is an intelligent, easy-to-use document-based knowledge retrieval and question-answering system. It allows users to search across PDFs and text files and receive detailed, human-like answers powered by advanced Large Language Models (LLMs). This project was developed as a placement project inspired by a real-world requirement from **Unthinkable Solutions**.

All operations in DocuMind are **in-memory**, meaning that documents and embeddings are never saved to disk. This ephemeral design ensures your data remains private, stateless, and easy to reset.

---

## Features

* **Multiple Document Uploads:** Upload PDFs or text files quickly via a simple web interface.  
* **In-Memory Embeddings:** Convert documents into vectors using **Google Gemini embeddings** for fast and accurate retrieval.  
* **Smart Chunking:** Automatically splits large documents into sentence-aware chunks for better contextual understanding.  
* **RAG-Based Answers:** Combines document knowledge with general information to provide precise, retrieval-augmented responses.  
* **Human-Friendly Responses:** Generates natural, readable answers with bullet points, paragraphs, and bold highlights.  
* **Automatic Cleanup:** Uploaded files are removed after processing, and FAISS memory is cleared on shutdown.  
* **Secure & Stateless:** No persistent storage—everything operates in-memory for safe testing and experimentation.  

---

## Tech Stack

* **Backend:** Python & Flask  
* **Embeddings:** Google Gemini API (`embedding-gecko-001`)  
* **Vector Store:** FAISS (in-memory)  
* **Frontend:** HTML, CSS, JavaScript for document upload and querying  
* **Environment Management:** `dotenv` for API keys  
* **Document Parsing:** PyPDF2 for PDFs, UTF-8 reading for TXT/MD files  

---

## How It Works

1. **Upload Documents:** Users upload PDFs or TXT/MD files via the web interface.  
2. **Text Extraction & Chunking:** Documents are parsed and split into smaller, sentence-aware chunks.  
3. **Embedding Generation:** Each chunk is converted into a vector using Google Gemini embeddings.  
4. **In-Memory Storage:** Vectors and metadata are stored in FAISS for rapid retrieval.  
5. **Querying:** Users ask a question, which is converted into an embedding and matched with the most relevant document chunks.  
6. **Answer Synthesis:** The LLM generates a detailed, human-friendly answer using retrieved chunks and general knowledge.  
7. **Automatic Cleanup:** Uploaded files are deleted, and memory is cleared after each session.  

---

## Demo Video

Watch the full demonstration of **DocuMind 1.0** here:  

[![Watch Demo](https://img.youtube.com/vi/-CFPwAQ85Ds/0.jpg)](https://youtu.be/-CFPwAQ85Ds)  

Click the image or the link to see the project in action on YouTube.  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/theindianarmytan05-create/DocuMind_1.0.git
cd DocuMind_1.0
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
GEMINI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=your_flask_secret
EMBEDDING_DIM=1536
```

4. Run the app:

```bash
python app.py
```

5. Open your browser and visit `http://localhost:5000` to upload documents and start querying.
```
