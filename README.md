# RAG Chatbot QnA

A Streamlit RAG (Retrieval-Augmented Generation) chatbot built with LangChain. Upload one or more PDFs, and it chunks and embeds them into an in-memory vector store, then uses a tool-calling agent to answer questions strictly from the retrieved document context.

## Features

- Multi-file PDF upload via a Streamlit UI (`PyPDFDirectoryLoader`)
- Text chunking with `RecursiveCharacterTextSplitter`
- In-memory vector storage and similarity search (`InMemoryVectorStore`)
- OpenAI embeddings (`text-embedding-3-large`)
- Tool-calling agent (`langgraph`/`create_agent`) that retrieves context via a `retrival_tool` before answering, with conversation memory per session (`InMemorySaver`)
- The agent is instructed to issue full, self-contained sub-questions (not bare keywords) to the retrieval tool, and to call it once per part of a multi-part question
- Source attribution — retrieved chunks are tagged with their originating file and page, e.g. `[Source: report.pdf, page 3]`, and the agent cites them in its answer
- Context-grounded answers — replies with "No data found in the provided document." when the answer isn't in the uploaded documents
- Shows the names of the files currently in use in the chat UI
- Automatically clears any leftover uploaded PDFs from previous sessions on a fresh app load

## Tech Stack

- Python 3.13+
- [LangChain](https://python.langchain.com/) / [LangGraph](https://www.langchain.com/langgraph) / LangChain Community / LangChain Core
- [Streamlit](https://streamlit.io/) for the chat UI
- OpenAI (chat + embeddings) via `langchain-openai`
- [uv](https://docs.astral.sh/uv/) for dependency management

## Project Structure

```
.
├── main.py       # Streamlit app: PDF upload, vector store, retrieval agent, and chat UI
├── llm.py        # LLM provider initialization (OpenAI / Groq)
├── files/        # Sample PDFs (e.g. sample1.pdf)
├── user_docs/    # PDFs uploaded via the Streamlit UI at runtime (cleared on fresh app load)
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

1. Upload one or more PDF files (max 1 MB each).
2. Once processing finishes, ask questions in the chat input.

The vector store, agent, and chat history live only in the Streamlit session — restarting the app or starting a new session clears them, and any PDFs left in `user_docs/` from a previous session are removed automatically.

## How It Works

1. Uploaded PDFs are saved to `user_docs/`, loaded with `PyPDFDirectoryLoader`, and split into overlapping chunks (1000 chars, 300 overlap).
2. Chunks are embedded (`text-embedding-3-large`) and stored in an in-memory vector store for the session.
3. A LangGraph agent is created with a `retrival_tool` that runs similarity search (`k=5`) against the vector store and returns chunks tagged with their source file and page.
4. On each question, the agent decides when and how to call `retrival_tool` — using full, self-contained sub-questions for multi-part questions — then answers using only the retrieved, source-tagged context, citing the file (and page) behind each claim. Conversation state is kept per session via `InMemorySaver`.
