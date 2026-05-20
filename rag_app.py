import os
import json
import hashlib
from datetime import datetime
from pathlib import Path  # used for _SYSTEM_FONTS font paths
import streamlit as st
import voyageai
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv
import tempfile

load_dotenv()

PERSONA_NAMES = ["Compliance Officer", "RegTech Product Manager", "Policy & Regulatory Affairs Analyst"]


class VoyageEmbeddings(Embeddings):
    def __init__(self, model: str = "voyage-3", api_key: str | None = None):
        self.client = voyageai.Client(api_key=api_key or os.environ["VOYAGE_API_KEY"])
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embed(texts, model=self.model, input_type="document")
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.client.embed([text], model=self.model, input_type="query")
        return result.embeddings[0]


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **_):
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, *_, **__):
        self.container.markdown(self.text)


def analyze_document(text: str) -> dict:
    """Single Haiku call: checks compliance relevance and detects jurisdictions."""
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=150,
    )
    response = llm.invoke(
        "Analyze this regulatory document excerpt. Return a JSON object with exactly these fields:\n"
        '- "is_compliance": true if the document is related to AML, financial crime compliance, '
        'or financial regulation; false otherwise\n'
        '- "jurisdictions": array of applicable jurisdictions detected from the content. '
        'Use values from: "US", "UK", "EU", "India", "Japan", "Global". '
        'Use "Global" for multi-jurisdictional documents like FATF guidance. '
        'A document may have multiple values.\n'
        '- "reason": one short sentence explaining your determination\n\n'
        "Return only valid JSON, no markdown or extra text.\n\n"
        f"Document excerpt:\n{text[:2000]}"
    )
    try:
        content = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(content)
    except Exception:
        is_comp = "compliance" in response.content.lower() or "aml" in response.content.lower()
        return {"is_compliance": is_comp, "jurisdictions": ["Unknown"], "reason": response.content[:120]}


def confidence_label(score: float) -> tuple[str, str]:
    """Returns (level, message) for a given average relevance score."""
    if score >= 0.75:
        return "high", ""
    if score >= 0.5:
        return "medium", "Moderate confidence — limited source coverage. Cross-check before relying on this answer."
    return "low", "Low confidence — answer may not be well-supported by your documents. Verify manually."


def export_chat_as_markdown(
    messages: list,
    persona: str,
    jurisdictions: list[str],
    indexed_docs: list,
) -> str:
    lines = [
        "# AML Regulatory Knowledge Base — Chat Export",
        "",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Persona:** {persona}  ",
        f"**Jurisdictions:** {', '.join(jurisdictions)}  ",
        "",
        "## Indexed Documents",
    ]
    for doc in indexed_docs:
        badges = " ".join(f"`{j}`" for j in doc.get("jurisdictions", []))
        lines.append(f"- {doc['name']} {badges}")
    lines += ["", "---", "", "## Conversation", ""]

    for msg in messages:
        if msg["role"] == "user":
            lines += [f"**You:** {msg['content']}", ""]
        else:
            lines += [f"**Assistant:** {msg['content']}", ""]
            level = msg.get("confidence_level", "")
            if level:
                score_pct = round(msg.get("confidence_score", 0) * 100)
                conf_line = f"Confidence: {score_pct}% · {level.capitalize()}"
                if msg.get("confidence_msg"):
                    conf_line += f" — {msg['confidence_msg']}"
                lines += [f"> _{conf_line}_", ""]
            if msg.get("sources"):
                lines.append("*Sources:*")
                for src in msg["sources"]:
                    lines.append(f"- {src}")
            lines += ["", "---", ""]

    return "\n".join(lines)


def export_chat_as_docx(messages: list, persona: str, jurisdictions: list[str], indexed_docs: list) -> bytes:
    from docx import Document
    from docx.shared import RGBColor
    from io import BytesIO

    doc = Document()
    doc.add_heading("AML Regulatory Knowledge Base — Chat Export", 0)

    meta = doc.add_paragraph()
    meta.add_run("Exported: ").bold = True
    meta.add_run(datetime.now().strftime("%Y-%m-%d %H:%M"))
    meta = doc.add_paragraph()
    meta.add_run("Persona: ").bold = True
    meta.add_run(persona)
    meta = doc.add_paragraph()
    meta.add_run("Jurisdictions: ").bold = True
    meta.add_run(", ".join(jurisdictions))

    doc.add_heading("Indexed Documents", 1)
    for d in indexed_docs:
        badges = ", ".join(d.get("jurisdictions", []))
        doc.add_paragraph(f"{d['name']}  [{badges}]", style="List Bullet")

    doc.add_heading("Conversation", 1)
    for msg in messages:
        if msg["role"] == "user":
            p = doc.add_paragraph()
            p.add_run("You: ").bold = True
            p.add_run(msg["content"])
        else:
            p = doc.add_paragraph()
            p.add_run("Assistant: ").bold = True
            p.add_run(msg["content"])
            if msg.get("confidence_level"):
                score_pct = round(msg.get("confidence_score", 0) * 100)
                level = msg["confidence_level"].capitalize()
                conf_text = f"Confidence: {score_pct}% · {level}"
                if msg.get("confidence_msg"):
                    conf_text += f" — {msg['confidence_msg']}"
                cp = doc.add_paragraph(conf_text)
                cp.runs[0].italic = True
                cp.runs[0].font.color.rgb = RGBColor(0x80, 0x80, 0x80)
            if msg.get("sources"):
                sp = doc.add_paragraph()
                sp.add_run("Sources: ").italic = True
                for src in msg["sources"]:
                    doc.add_paragraph(src, style="List Bullet")
        doc.add_paragraph()

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


_SYSTEM_FONTS = {
    "": Path("C:/Windows/Fonts/arial.ttf"),
    "B": Path("C:/Windows/Fonts/arialbd.ttf"),
    "I": Path("C:/Windows/Fonts/ariali.ttf"),
}


def export_chat_as_pdf(messages: list, persona: str, jurisdictions: list[str], indexed_docs: list) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for style, path in _SYSTEM_FONTS.items():
        pdf.add_font("Arial", style=style, fname=str(path))

    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, "AML Regulatory Knowledge Base — Chat Export")
    pdf.ln(2)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 6, f"Persona: {persona}", ln=True)
    pdf.cell(0, 6, f"Jurisdictions: {', '.join(jurisdictions)}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Indexed Documents", ln=True)
    pdf.set_font("Arial", "", 10)
    for d in indexed_docs:
        badges = ", ".join(d.get("jurisdictions", []))
        pdf.cell(0, 6, f"  - {d['name']}  [{badges}]", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Conversation", ln=True)
    pdf.ln(2)

    for msg in messages:
        if msg["role"] == "user":
            pdf.set_font("Arial", "B", 10)
            pdf.multi_cell(0, 6, f"You:  {msg['content']}")
        else:
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, "Assistant:", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 6, msg["content"])
            if msg.get("confidence_level"):
                score_pct = round(msg.get("confidence_score", 0) * 100)
                level = msg["confidence_level"].capitalize()
                conf_text = f"Confidence: {score_pct}% - {level}"
                if msg.get("confidence_msg"):
                    conf_text += f" - {msg['confidence_msg']}"
                pdf.set_font("Arial", "I", 9)
                pdf.multi_cell(0, 5, conf_text)
            if msg.get("sources"):
                pdf.set_font("Arial", "I", 9)
                pdf.cell(0, 5, "Sources:", ln=True)
                for src in msg["sources"]:
                    pdf.multi_cell(0, 5, f"  - {src}")
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.l_margin, pdf.get_y() + 3, pdf.w - pdf.r_margin, pdf.get_y() + 3)
        pdf.ln(7)

    return bytes(pdf.output())


def get_persona_prompt(persona: str, jurisdictions: list[str]) -> PromptTemplate:
    jur = ", ".join(jurisdictions) if jurisdictions else "the available jurisdictions"
    templates = {
        "Compliance Officer": f"""You are an AML compliance expert advising a compliance officer at a financial institution.
The knowledge base covers the following jurisdictions: {jur}.
Focus on practical obligations, examination risk, and what regulators expect in practice.
Be direct. Cite specific thresholds, timeframes, and trigger criteria where they exist.
Flag where non-compliance creates material regulatory or reputational risk.
For cross-jurisdictional questions, explicitly compare and contrast requirements.

Context from regulatory documents:
{{context}}

Question: {{question}}
Answer:""",

        "RegTech Product Manager": f"""You are an AML regulatory expert advising a product manager building compliance systems.
The knowledge base covers the following jurisdictions: {jur}.
Translate regulatory requirements into product specifications — data fields, workflow logic, thresholds, audit trail requirements.
Structure answers as: requirement → what to build → edge cases to handle.
Avoid regulatory jargon; use systems and product language.
For cross-jurisdictional questions, highlight where requirements diverge and where a single implementation can satisfy multiple regimes.

Context from regulatory documents:
{{context}}

Question: {{question}}
Answer:""",

        "Policy & Regulatory Affairs Analyst": f"""You are an AML regulatory expert advising a policy and regulatory affairs analyst.
The knowledge base covers the following jurisdictions: {jur}.
Focus on regulatory intent, jurisdictional comparisons, interpretive nuance, and gaps in the framework.
Use precise regulatory language. Note where rules are ambiguous or where guidance diverges from legislation.
Reference cross-jurisdictional parallels and cite section numbers where relevant.

Context from regulatory documents:
{{context}}

Question: {{question}}
Answer:""",
    }
    return PromptTemplate(input_variables=["context", "question"], template=templates[persona])


st.set_page_config(page_title="AML Knowledge Base", page_icon="⚖️", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.header("Documents")
    st.caption("Upload English-language PDFs. Jurisdiction is auto-detected.")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        content_hash = hashlib.md5()
        for f in sorted(uploaded_files, key=lambda x: x.name):
            content_hash.update(f.name.encode())
            content_hash.update(f.getvalue())
        file_hash = content_hash.hexdigest()

        if st.session_state.get("indexed_hash") != file_hash:
            all_docs = []
            accepted = []

            with st.status("Processing documents...", expanded=True) as status:
                for uf in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.getvalue())
                        tmp_path = tmp.name
                    docs = PyPDFLoader(tmp_path).load()

                    st.write(f"Analysing **{uf.name}**...")
                    sample = "\n\n".join(d.page_content for d in docs[:2])
                    result = analyze_document(sample)

                    if not result["is_compliance"]:
                        reason = result.get("reason", "")
                        st.warning(f"**{uf.name}** skipped — {reason}")
                        continue

                    jurisdictions = result.get("jurisdictions") or ["Unknown"]
                    jur_str = ", ".join(jurisdictions)
                    for doc in docs:
                        doc.metadata["jurisdiction"] = jur_str
                    all_docs.extend(docs)
                    accepted.append({"name": uf.name, "jurisdictions": jurisdictions})
                    st.write(f"✓ **{uf.name}** — detected: `{jur_str}`")

                if not all_docs:
                    status.update(label="No compliance documents found.", state="error")
                else:
                    chunks = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    ).split_documents(all_docs)

                    embeddings = VoyageEmbeddings(model="voyage-3")
                    st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
                    st.session_state.indexed_hash = file_hash
                    st.session_state.indexed_docs = accepted
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    status.update(
                        label=f"Ready — {len(chunks)} chunks from {len(accepted)} document(s).",
                        state="complete",
                    )

    if st.session_state.get("indexed_docs"):
        st.markdown("**Indexed documents:**")
        for doc in st.session_state.indexed_docs:
            badges = " ".join(f"`{j}`" for j in doc["jurisdictions"])
            st.markdown(f"- {doc['name']} &nbsp; {badges}")

    st.divider()

    st.header("Persona")
    st.selectbox("I am a...", PERSONA_NAMES, key="persona")

# --- Main area ---
st.title("AML Regulatory Knowledge Base")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.info("Upload documents in the sidebar to get started.")
else:
    active_jurisdictions = list({
        j
        for doc in st.session_state.get("indexed_docs", [])
        for j in doc.get("jurisdictions", [])
    })

    col_title, col_export, col_clear = st.columns([3, 1.2, 1.2])
    with col_title:
        st.subheader("Chat")
    with col_export:
        has_messages = bool(st.session_state.messages)
        with st.popover("Export chat", disabled=not has_messages, use_container_width=True):
            fmt = st.radio(
                "Format",
                ["Markdown (.md)", "Word (.docx)", "PDF (.pdf)"],
                label_visibility="collapsed",
            )
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            args = (
                st.session_state.messages,
                st.session_state.get("persona", PERSONA_NAMES[0]),
                active_jurisdictions,
                st.session_state.get("indexed_docs", []),
            )
            if fmt == "Markdown (.md)":
                data, fname, mime = export_chat_as_markdown(*args), f"aml_chat_{ts}.md", "text/markdown"
            elif fmt == "Word (.docx)":
                data = export_chat_as_docx(*args)
                fname = f"aml_chat_{ts}.docx"
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else:
                data, fname, mime = export_chat_as_pdf(*args), f"aml_chat_{ts}.pdf", "application/pdf"
            st.download_button("Download", data=data, file_name=fname, mime=mime, use_container_width=True)
    with col_clear:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    if not st.session_state.messages:
        jur_str = ", ".join(active_jurisdictions)
        persona = st.session_state.get("persona", PERSONA_NAMES[0])
        st.markdown(
            f"Knowledge base ready — **{jur_str}** documents indexed. "
            f"Responding as **{persona}**. Ask me anything."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("confidence_level"):
                score_pct = round(msg.get("confidence_score", 0) * 100)
                level = msg["confidence_level"].capitalize()
                conf_text = f"Confidence: {score_pct}% · {level}"
                if msg.get("confidence_msg"):
                    conf_text += f" — {msg['confidence_msg']}"
                if msg["confidence_level"] == "low":
                    st.warning(conf_text)
                elif msg["confidence_level"] == "medium":
                    st.caption(conf_text)
                else:
                    st.caption(conf_text)
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.caption(src)

    if question := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            stream_handler = StreamHandler(placeholder)

            llm = ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=os.environ["ANTHROPIC_API_KEY"],
                streaming=True,
                callbacks=[stream_handler],
            )
            llm_condense = ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )

            persona = st.session_state.get("persona", PERSONA_NAMES[0])
            prompt = get_persona_prompt(persona, active_jurisdictions)

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                condense_question_llm=llm_condense,
                retriever=st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt},
            )

            result = qa_chain.invoke({
                "question": question,
                "chat_history": st.session_state.chat_history,
            })

            sources = [
                f"Page {doc.metadata.get('page', '?')} — "
                f"{os.path.basename(doc.metadata.get('source', 'Unknown'))} "
                f"[{doc.metadata.get('jurisdiction', '?')}]"
                for doc in result.get("source_documents", [])
            ]

            # Confidence scoring from retrieval similarity
            scored = st.session_state.vectorstore.similarity_search_with_relevance_scores(question, k=5)
            avg_score = sum(s for _, s in scored) / len(scored) if scored else 0
            conf_level, conf_msg = confidence_label(avg_score)

            score_pct = round(avg_score * 100)
            level_display = conf_level.capitalize()
            conf_text = f"Confidence: {score_pct}% · {level_display}"
            if conf_msg:
                conf_text += f" — {conf_msg}"
            if conf_level == "low":
                st.warning(conf_text)
            elif conf_level == "medium":
                st.caption(conf_text)
            else:
                st.caption(conf_text)

            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.caption(src)

        answer = stream_handler.text
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confidence_score": avg_score,
            "confidence_level": conf_level,
            "confidence_msg": conf_msg,
        })
        st.session_state.chat_history.append((question, answer))
