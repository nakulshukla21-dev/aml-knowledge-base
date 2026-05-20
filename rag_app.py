import os
import streamlit as st
import voyageai
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import tempfile


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

load_dotenv()

st.title("AML Regulatory Knowledge Base")
st.write("Upload regulatory documents and ask questions across all of them.")

# File uploader - allows multiple files
uploaded_files = st.file_uploader(
    "Upload PDF documents (FATF, FinCEN, OFAC etc.)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Build Knowledge Base", use_container_width=True):
            with st.spinner("Processing documents..."):

                # Save uploaded files temporarily and load them
                all_docs = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path)
                    all_docs.extend(loader.load())

                # Split documents into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(all_docs)

                # Create vector store
                embeddings = VoyageEmbeddings(model="voyage-3")
                vectorstore = Chroma.from_documents(chunks, embeddings)
                st.session_state.vectorstore = vectorstore
                st.success(f"Knowledge base built from {len(uploaded_files)} documents — {len(chunks)} chunks indexed.")

    with col2:
        if st.button("Summarize Documents", use_container_width=True):
            llm = ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=os.environ["ANTHROPIC_API_KEY"],
                max_tokens=4096,
            )
            st.subheader("Summaries")
            for uploaded_file in uploaded_files:
                with st.spinner(f"Summarizing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    pages = PyPDFLoader(tmp_path).load()
                    full_text = "\n\n".join(p.page_content for p in pages)

                    prompt = (
                        "You are summarizing an AML/regulatory document. Produce a structured "
                        "summary covering: (1) the document's purpose and scope, (2) key definitions, "
                        "(3) main requirements and obligations, (4) thresholds, deadlines, or numeric "
                        "criteria, and (5) penalties or enforcement provisions. Be specific — cite "
                        "section numbers where relevant.\n\n"
                        f"Document ({len(pages)} pages):\n\n{full_text}"
                    )
                    response = llm.invoke(prompt)

                st.markdown(f"### {uploaded_file.name}")
                st.write(response.content)

# Q&A interface
if "vectorstore" in st.session_state:
    st.subheader("Ask a Question")
    question = st.text_input("What would you like to know?")

    if question:
        with st.spinner("Searching documents..."):
            llm = ChatAnthropic(
                model="claude-sonnet-4-6",
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )
            result = qa_chain.invoke({"query": question})

        st.subheader("Answer")
        st.write(result["result"])

        st.subheader("Sources")
        for doc in result["source_documents"]:
            st.caption(f"Page {doc.metadata.get('page', '?')} — {doc.metadata.get('source', 'Unknown')}")