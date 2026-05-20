# AML Regulatory Knowledge Base

An AI-powered chatbot for querying AML/financial crime regulatory documents using RAG (Retrieval Augmented Generation). Upload documents from multiple jurisdictions, select your professional persona, and have a multi-turn conversation across your entire document set.

## Features

- **Multi-turn chat** — Conversational interface with full session history. Follow-up questions understand prior context.
- **Auto compliance check** — Each uploaded document is screened by Claude Haiku before indexing. Non-AML documents are rejected with a reason.
- **Auto jurisdiction detection** — Jurisdiction(s) are detected from document content (US, UK, EU, India, Japan, Global). A single document can span multiple jurisdictions.
- **3 professional personas** — Answers are tailored to your role:
  - *Compliance Officer* — practical obligations, thresholds, examination risk
  - *RegTech Product Manager* — product specs, data fields, workflow logic, edge cases
  - *Policy & Regulatory Affairs Analyst* — regulatory intent, cross-jurisdictional comparisons, interpretive nuance
- **Cross-jurisdictional queries** — Ask questions across documents from different jurisdictions in a single query
- **Confidence scoring** — Every answer shows a relevance score (%) and High/Medium/Low classification based on how well the retrieved chunks support the answer
- **Streaming responses** — Answers stream word-by-word in real time
- **Multi-format export** — Export the full chat as Markdown, Word (.docx), or PDF — includes metadata, confidence scores, and sources per answer

## Tech Stack

| Component | Detail |
|---|---|
| UI | Streamlit |
| LLM | Claude Sonnet 4.6 (`claude-sonnet-4-6`) |
| Compliance & jurisdiction check | Claude Haiku 4.5 (`claude-haiku-4-5`) |
| Question rephrasing | Claude Haiku 4.5 (fast, cheap condense step) |
| Embeddings | Voyage AI (`voyage-3`) |
| Vector store | ChromaDB (in-memory, per session) |
| Orchestration | LangChain (`ConversationalRetrievalChain`) |
| PDF export | fpdf2 + system Arial font |
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
   streamlit run rag_app.py
   ```

## Usage

1. **Upload PDFs** in the sidebar — English-language AML/regulatory documents only
2. The app automatically screens each document for compliance relevance and detects its jurisdiction(s)
3. **Select a persona** in the sidebar to tailor the style and framing of answers
4. **Chat** — ask questions across all indexed documents. Cross-jurisdictional questions work out of the box
5. **Export** the conversation via the Export chat button (Markdown / Word / PDF)

## Notes

- Session state is in-memory — documents and chat history reset on page refresh (Streamlit limitation)
- PDF export uses Windows Arial font; Linux deployments may need font path adjustments
- English documents only — no translation or multi-language support

## Security

API keys are loaded from `.env` and never committed. The `.gitignore` excludes all `.env` files and the local ChromaDB store.
