# AML Regulatory Knowledge Base

An AI-powered chatbot for querying AML/financial crime regulatory documents using RAG (Retrieval Augmented Generation). Upload documents from multiple jurisdictions, select your professional persona, and have a multi-turn conversation across your entire document set.

## Live Demo
[Try it here](https://bit.ly/aml-knowledge-base)

## Features

- **Multi-turn chat** — Conversational interface with full session history. Follow-up questions are rewritten into standalone search queries using recent context.
- **Auto compliance check** — Each uploaded document is screened by Claude Haiku before indexing. Non-AML documents are rejected with a reason.
- **Auto jurisdiction detection** — Jurisdiction(s) are detected from document content (US, UK, EU, India, Japan, Global). A single document can span multiple jurisdictions.
- **Institution detection** — Institution name, document title, and document type are extracted at upload and shown in the sidebar.
- **Document header indexing** — Each PDF gets a dedicated summary chunk with the opening pages, so names and titles near the start of a document are always available to retrieval.
- **3 professional personas** — Answers are tailored to your role:
  - *Compliance Officer* — practical obligations, thresholds, examination risk
  - *RegTech Product Manager* — product specs, data fields, workflow logic, edge cases
  - *Policy & Regulatory Affairs Analyst* — regulatory intent, cross-jurisdictional comparisons, interpretive nuance
- **Cross-jurisdictional queries** — Ask questions across documents from different jurisdictions in a single query
- **Grounded answers** — Prompts instruct the model to answer only from retrieved document excerpts and to say when the context is insufficient
- **Confidence scoring** — After each answer, Haiku evaluates how well the response is supported by the retrieved excerpts, returning a score (%), High/Medium/Low label, and a short explanation
- **Streaming responses** — Answers stream word-by-word in real time
- **In-app guide** — Collapsible “How to use this tool” section on the main page
- **Chat export** — Export the full conversation as Markdown or Word (.docx), including metadata, confidence scores, and sources per answer

## How it works

```text
Upload PDFs
  → screen for AML relevance + detect jurisdiction/institution (Haiku)
  → split into header + body chunks
  → embed with Voyage and store in in-memory ChromaDB

Ask a question
  → rewrite follow-ups into standalone queries (Haiku)
  → retrieve header chunks + top matching body chunks
  → generate answer from retrieved context (Sonnet, streamed)
  → evaluate grounding vs excerpts (Haiku)
  → show answer, confidence reason, and sources
```

## Tech Stack

| Component | Detail |
|---|---|
| UI | Streamlit |
| Answer generation | Claude Sonnet 4.6 (`claude-sonnet-4-6`) |
| Document screening, institution extraction, question condensing, confidence evaluation | Claude Haiku 4.5 (`claude-haiku-4-5`) |
| Embeddings | Voyage AI (`voyage-3`) |
| Vector store | ChromaDB (in-memory, per session) |
| PDF loading | LangChain `PyPDFLoader` + `pypdf` |
| Orchestration | LangChain (custom retrieve → format → answer flow) |
| Word export | python-docx |

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
   pip install -r requirements.txt
   ```

4. **Set API keys** — create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   VOYAGE_API_KEY=your_voyage_api_key
   ```

5. **Run the app**
   ```bash
   python -m streamlit run rag_app.py
   ```

## Usage

1. **Upload PDFs** in the sidebar — English-language AML/regulatory documents only
2. **Remove old files** from the uploader before loading a new document set — all selected PDFs form one active knowledge base
3. Confirm the sidebar shows each indexed document with its jurisdiction and detected institution name
4. **Select a persona** in the sidebar to tailor the style and framing of answers
5. **Chat** — ask questions across all indexed documents. Each answer shows sources and a confidence score with an explanation
6. Use **Reset knowledge base** when switching to a completely new document set
7. **Export** the conversation via the Export chat button (Markdown / Word)

## Session behavior

| Action | What resets |
|---|---|
| Page refresh | Documents, vector index, chat history |
| Change uploaded PDF set | Re-indexes documents and clears chat |
| Remove all files from uploader | Clears the knowledge base |
| Clear chat | Chat only — documents stay indexed |
| Reset knowledge base | Documents, index, and chat |

## Deploying on Streamlit Community Cloud

1. Deploy from your GitHub repo with entrypoint `rag_app.py`
2. **Use Python 3.12** — ChromaDB does not work on Python 3.14. In **Advanced settings** at deploy time, select **3.12**. If the app was already deployed on 3.14, delete and redeploy (Python version cannot be changed after deploy).
3. Add secrets in **Advanced settings → Secrets** (or app Settings → Secrets):
   ```toml
   ANTHROPIC_API_KEY = "your_key"
   VOYAGE_API_KEY = "your_key"
   ```

## Notes

- Session state is in-memory — documents and chat history reset on page refresh (Streamlit limitation)
- Text-based PDFs work best; scanned/image PDFs may not extract cleanly
- English documents only — no translation or multi-language support
- PDF export exists in code but is currently disabled in the UI

## Security

API keys are loaded from `.env` and never committed. The `.gitignore` excludes all `.env` files and the local ChromaDB store.
