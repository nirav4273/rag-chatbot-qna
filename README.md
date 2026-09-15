# RAG Chatbot QnA

A command-line RAG (Retrieval-Augmented Generation) chatbot built with LangChain and LangGraph. It loads a PDF, chunks and embeds it into a local Chroma vector store, and answers questions strictly from the retrieved document context.

## Features

- PDF ingestion via `PyPDFLoader`
- Text chunking with `RecursiveCharacterTextSplitter`
- Vector storage and similarity search using Chroma (persisted locally)
- OpenAI embeddings (`text-embedding-3-large`)
- Conversational agent (LangGraph `create_agent`) with in-memory checkpointing for session state
- Context-grounded answers — responds with "No data found" when the answer isn't in the document

## Tech Stack

- Python 3.13+
- [LangChain](https://python.langchain.com/) / LangChain Community / LangChain Core
- [LangGraph](https://langchain-ai.github.io/langgraph/) for the agent runtime
- [Chroma](https://www.trychroma.com/) as the vector store
- OpenAI (chat + embeddings) via `langchain-openai`
- [uv](https://docs.astral.sh/uv/) for dependency management

## Project Structure

```
.
├── main.py       # Entry point: loads PDF, builds vector store, runs the Q&A loop
├── llm.py        # LLM provider initialization (OpenAI / Groq)
├── files/        # Source PDFs used for retrieval (e.g. sample1.pdf)
├── chroma_langchain_db/  # Local persisted Chroma vector store
└── pyproject.toml
```

## Setup

1. Install dependencies with [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync
   ```

2. Create a `.env` file in the project root with the following variables:
   ```
   MODEL_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-4o-mini
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL=your_groq_model
   ```
   > Note: `OPENAI_API_KEY` is required for embeddings regardless of the selected `MODEL_PROVIDER`.

3. Place the PDF you want to query in the `files/` directory (default: `files/sample1.pdf`).

## Usage

Run the chatbot:

```bash
uv run main.py
```

You'll be prompted to ask questions about the loaded PDF:

```
Ask question from PDF: What is this document about?
ANS --->  ...
```

Type `exit` to quit.

## How It Works

1. The PDF is loaded and split into overlapping chunks (500 chars, 150 overlap).
2. Chunks are embedded and stored in a persistent Chroma vector store.
3. On each question, the most relevant chunks are retrieved via similarity search.
4. The retrieved context and question are rendered into a prompt and passed to the LangGraph agent, which answers using only that context.
