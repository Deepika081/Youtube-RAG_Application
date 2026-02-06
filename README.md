# YouTube Video Q&A Chatbot (RAG-based)

This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about a YouTube video and continue the conversation like a chat.

The system:
- Extracts the YouTube video transcript
- Chunks and embeds the transcript
- Stores embeddings in a vector database (ChromaDB)
- Retrieves relevant context for each question
- Uses an LLM (Groq + LLaMA) to generate accurate answers
- Maintains chat history for follow-up questions

---

## Features

- Ask questions about any YouTube video (with transcripts enabled)
- Context-aware answers using RAG
- Chat-style follow-up questions supported
- Automatic transcript caching per video
- FastAPI backend
- Gradio-based chatbot UI

---

## Tech Stack

- Python 3.12
- FastAPI
- Gradio
- LangChain
- ChromaDB
- YouTube Transcript API
- Groq API (LLaMA model)

---

## Project Structure

Youtube_RAG_App/
│
├── api/
|   └── __init__.py
│   └── main.py             # FastAPI backend
│
├── app/
|   └── __init__.py
│   └── rag_app.py          # Core RAG + chatbot logic
│
├── ui/     
|   └── __init__.py       
│   └── gradio_app.py       # Gradio chatbot UI
|
├── requirements.txt
|
└── README.md

---

## Setup Instructions

### 1. Create Virtual Environment (Python 3.12)

python -m venv venv  
venv\Scripts\activate  

---

### 2. Install Dependencies

pip install -r requirements.txt

---

### 3. Environment Variables

Create a `.env` file in the root directory:

GROQ_API_KEY=your_groq_api_key_here

---

## Running the Application

### Option 1: Run Gradio Chatbot

python -m ui.gradio_app

This will launch a browser-based chatbot UI.

---

### Option 2: Run FastAPI Backend

uvicorn api.main:app --reload

Open:
http://127.0.0.1:8000/docs

---

## How It Works

1. User provides a YouTube video URL and a question
2. Video ID is extracted
3. If embeddings already exist:
   - Relevant chunks are retrieved
4. If embeddings do not exist:
   - Transcript is fetched
   - Text is chunked
   - Embeddings are created and stored
5. Relevant context is retrieved
6. LLM generates an answer
7. Chat history is used for follow-up questions

---

## Limitations

- Works only for videos with English transcripts
- No authentication or rate limiting
- Prototype-level implementation

---

## Future Improvements

- Multi-language transcript support
- Persistent vector storage
- Streaming responses
- User session management
- Improved chunking strategies

---

## Status

This project is a working prototype built to explore RAG pipelines, vector databases, and LLM-powered chat systems.