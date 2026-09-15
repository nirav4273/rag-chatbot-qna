# RAG Chatbot QnA

A Streamlit RAG (Retrieval-Augmented Generation) chatbot built with LangChain. Upload one or more PDFs, and it chunks and embeds them into an in-memory vector store, then answers questions strictly from the retrieved document context.

## Features

- Multi-file PDF upload via a Streamlit UI (`PyPDFDirectoryLoader`)
- Text chunking with `RecursiveCharacterTextSplitter`
- In-memory vector storage and similarity search (`InMemoryVectorStore`)
- OpenAI embeddings (`text-embedding-3-large`)
- Query decomposition: each question is first split by the LLM into self-contained sub-questions, each retrieved separately, then answered in one final grounded LLM call
- Source attribution — answers cite the originating file and page, e.g. `[Source: report.pdf, page 3]`
- Context-grounded answers — responds with "No data found in the provided document." when the answer isn't in the uploaded documents

## Tech Stack

- Python 3.13+
- [LangChain](https://python.langchain.com/) / LangChain Community / LangChain Core
- [Streamlit](https://streamlit.io/) for the chat UI
- OpenAI (chat + embeddings) via `langchain-openai`
- [uv](https://docs.astral.sh/uv/) for dependency management

## Project Structure

```
.
├── main.py       # Streamlit app: PDF upload, vector store, query splitting, retrieval, and answering
├── llm.py        # LLM provider initialization (OpenAI / Groq)
├── files/        # Sample PDFs (e.g. sample1.pdf)
├── user_docs/    # PDFs uploaded via the Streamlit UI at runtime
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

## Usage

Run the Streamlit app:

```bash
uv run streamlit run main.py
```

Then, in the browser tab that opens:

1. Upload one or more PDF files.
2. Once processing finishes, ask questions in the chat input.

The vector store and chat history live only in the Streamlit session — restarting the app or re-uploading clears them.

## How It Works

1. Uploaded PDFs are saved to `user_docs/`, loaded, and split into overlapping chunks (1000 chars, 300 overlap).
2. Chunks are embedded and stored in an in-memory vector store for the session.
3. On each question:
   - The LLM breaks the question into a list of self-contained, detailed sub-questions.
   - Each sub-question is run through similarity search against the vector store, and the results are merged (deduplicated).
   - The combined, source-tagged context and the original question are passed to a final LLM call, which answers using only that context and cites sources.
