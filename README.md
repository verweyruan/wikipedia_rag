# Wikipedia RAG Assistant

A conversational AI app that answers your questions based on real Wikipedia content. Instead of relying on what the AI already knows, it reads the actual Wikipedia pages and answers based on what's written there.

---

## Learning Context

This project was built as part of my self-directed full stack learning journey.

Credits to **NeuralNine** on YouTube for explaining RAG so clearly and making it easy to understand.

Original tutorial: [Wikipedia RAG System - NeuralNine](https://www.youtube.com/watch?v=M9GtHb32F8w&list=PL7yh-TELLS1G9mmnBN3ZSY8hYgJ5kBOg-&index=44)

**What I changed from the tutorial:**
- Swapped OpenAI for **Groq** (free tier, faster inference) because I had already used OpenAI in a previous project and wanted to learn something new
- Swapped OpenAI embeddings for **HuggingFace embeddings** which run locally on your machine — no API calls, no cost
- Changed the Wikipedia topics to ones relevant to my own interests

---

## Features

- Ask any question and get answers based on real Wikipedia content
- Shows the exact Wikipedia chunks used to generate the answer — fully transparent
- Builds the index once and caches it locally so subsequent queries are fast
- Free to run — no OpenAI account or payment needed

---

## How RAG Works

RAG stands for Retrieval Augmented Generation. The idea is simple — instead of asking an AI what it already knows, you give it your own documents to read first.

Here's what happens under the hood:

1. The app fetches the Wikipedia pages you specify
2. Breaks them into smaller chunks
3. Converts each chunk into a vector embedding — a list of numbers that captures the meaning of that text
4. Stores all those vectors in a local index
5. When you ask a question it converts your question into a vector too
6. Finds the 3 most similar chunks from the index
7. Feeds those chunks to the LLM as context
8. The LLM answers based on the actual Wikipedia content — not its training data

This is the foundation of every "chat with your documents" product you've ever seen.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| RAG Framework | LlamaIndex |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace (BAAI/bge-small-en-v1.5) |
| Data Source | Wikipedia via LlamaIndex Wikipedia Reader |

---

## Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/verweyruan/wikipedia_rag.git
cd wikipedia_rag
```

**2. Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Add your free Groq API key — get one at console.groq.com, no card needed.

**5. Run the app**
```bash
streamlit run main.py
```

The first run will download the HuggingFace embedding model (~130MB) and fetch the Wikipedia pages. After that it loads from the local cache instantly.

---

## Example Questions

- *"What is machine learning?"*
- *"How is AI different from machine learning?"*
- *"What does a software engineer do?"*
- *"What is a Software as a Service business model?"*

---

## What I Learned

- How RAG works at a fundamental level — chunking, embeddings, vector search, and retrieval
- How LlamaIndex orchestrates the entire RAG pipeline with minimal code
- The difference between an LLM and an embedding model and why you need both
- How to use Groq as a free alternative to OpenAI for LLM inference
- How to use HuggingFace embeddings locally instead of paying for an API

---

## License

MIT License — free to use and modify for personal use.