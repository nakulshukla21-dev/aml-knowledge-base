# AML Regulatory Knowledge Base

An AI-powered chatbot that enables natural language Q&A across multiple AML/KYC regulatory documents using RAG (Retrieval Augmented Generation).

## What it does
Upload multiple regulatory PDFs (FATF recommendations, FinCEN advisories, OFAC guidance) and ask questions across all of them simultaneously. The system retrieves the most relevant sections and generates accurate, sourced answers using Claude.

## Why it matters
Compliance teams spend significant time manually searching across multiple regulatory documents. This tool demonstrates how RAG-based AI can dramatically accelerate regulatory research in financial institutions.

## Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| LLM | Claude Sonnet (`claude-sonnet-4-6`) via LangChain Anthropic |
| Embeddings | Voyage AI (`voyage-3`) |
| Vector Store | ChromaDB |
| PDF Loader | LangChain PyPDFLoader |

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/nakulshukla21-dev/aml-knowledge-base.git
   cd aml-knowledge-base
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit voyageai langchain langchain-anthropic langchain-community langchain-chroma langchain-text-splitters langchain-classic pypdf python-dotenv chromadb
   ```

4. **Set API keys** — create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   VOYAGE_API_KEY=your_voyage_api_key
   ```

5. **Run the app**
   ```bash
   streamlit run rag_app.py
   ```

## Usage

1. Upload one or more regulatory PDFs using the file uploader
2. Click **Build Knowledge Base** to chunk and index the documents
3. (Optional) Click **Summarize Documents** for a structured summary of each file
4. Once the knowledge base is built, type a question in the Q&A box to query across all documents

## Security

API keys are loaded from a `.env` file and never committed. The `.gitignore` excludes all `.env` files and the local ChromaDB store.
