"""DocMind RAG - Streamlit UI.   Run:  streamlit run app.py"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from docmind.evaluation import evaluate, load_questions
from docmind.llm import DEFAULT_MODELS, KEY_NAMES, PROVIDERS
from docmind.loader import SUPPORTED_EXTENSIONS
from docmind.pipeline import RAGPipeline
from docmind.textutils import highlight_html

st.set_page_config(page_title="DocMind RAG", page_icon="📄", layout="wide")

st.markdown(
    """<style>
    mark {background:#fde68a; color:#111; padding:0 2px; border-radius:3px;}
    .src-text {font-size:0.9rem; line-height:1.5;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models... (first run downloads ~90 MB, please wait)")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipe = get_pipeline()
cfg = pipe.settings

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📄 DocMind RAG")
    st.caption(f"Embeddings: {pipe.embedder.backend_name}")
    if not pipe.embedder.is_semantic:
        st.warning(
            "Semantic model not loaded (offline or not installed) - using the keyword-like fallback. "
            "Run `pip install sentence-transformers` and restart for best results.",
            icon="⚠️",
        )

    st.subheader("1. Answer engine")
    default_idx = PROVIDERS.index(cfg.llm_provider) if cfg.llm_provider in PROVIDERS else 0
    provider = st.selectbox(
        "LLM provider", PROVIDERS, index=default_idx,
        help="'extractive' needs no API key or model - it quotes the best sentences from your documents.",
    )
    model = ""
    if provider != "extractive":
        model = st.text_input("Model", value=os.getenv(f"{provider.upper()}_MODEL", DEFAULT_MODELS[provider]))
        if provider in KEY_NAMES:
            key = st.text_input(f"{KEY_NAMES[provider]}", type="password", value=os.getenv(KEY_NAMES[provider], ""))
            if key:
                os.environ[KEY_NAMES[provider]] = key  # kept in this process only
        else:
            st.caption("Ollama must be running locally: `ollama serve` and `ollama pull llama3.2`")

    st.subheader("2. Retrieval")
    top_k = st.slider("Passages to use (top-k)", 1, 8, cfg.top_k)
    hybrid = st.toggle("Hybrid search (meaning + keywords)", value=True)
    rerank = st.toggle("Cross-encoder re-ranking", value=False, help="More accurate, slower. Downloads a small model the first time.")
    min_conf = st.slider("Refuse below confidence", 0.0, 0.9, cfg.min_confidence, 0.05,
                         help="If the best passage scores below this, DocMind says 'not found' instead of guessing.")

    st.subheader("3. Documents")
    uploads = st.file_uploader("Upload PDF / TXT / MD", type=[e.strip(".") for e in SUPPORTED_EXTENSIONS], accept_multiple_files=True)
    chunk_size = st.select_slider("Chunk size (characters)", options=[300, 500, 800, 1000, 1200, 1500], value=cfg.chunk_size if cfg.chunk_size in (300, 500, 800, 1000, 1200, 1500) else 800)

    c1, c2 = st.columns(2)
    if c1.button("Index uploads", disabled=not uploads, use_container_width=True):
        cfg.upload_dir.mkdir(parents=True, exist_ok=True)
        bar = st.progress(0.0)
        for i, up in enumerate(uploads):
            target = cfg.upload_dir / Path(up.name).name
            target.write_bytes(up.getbuffer())
            rep = pipe.ingest_file(target, chunk_size=chunk_size, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
            st.toast(f"{rep['file']}: {rep['status']} ({rep['chunks']} chunks)")
        bar.empty()
        st.rerun()
    if c2.button("Load sample", use_container_width=True, help="Indexes the demo Acme HR policy PDF"):
        for rep in pipe.ingest_folder(cfg.sample_dir, chunk_size=chunk_size):
            st.toast(f"{rep['file']}: {rep['status']}")
        st.rerun()

    docs = pipe.list_documents()
    selected: list[str] = []
    if docs:
        st.caption("Tick the documents to search in:")
    for d in docs:
        a, b = st.columns([5, 1])
        if a.checkbox(f"{d['source']} · {d['pages']}p · {d['chunks']} chunks", value=True, key=f"sel_{d['doc_id']}"):
            selected.append(d["doc_id"])
        if b.button("🗑", key=f"del_{d['doc_id']}", help="Remove this document"):
            pipe.remove_document(d["doc_id"])
            st.rerun()
    doc_filter = None if len(selected) == len(docs) else selected

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------- rendering
def render_details(ans, query: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{ans.confidence:.0%}", help="How relevant the best retrieved passage is to your question.")
    c2.metric("Groundedness", ans.grounded_label, help="How well each answer sentence is supported by the retrieved text.")
    c3.metric("Latency", f"{ans.latency_ms} ms")
    c4.metric("Engine", ans.provider)
    if ans.note:
        st.info(ans.note)
    if ans.standalone_question and ans.standalone_question != query:
        st.caption(f"🔁 Understood your follow-up as: *{ans.standalone_question}*")
    if ans.unsupported:
        with st.expander(f"⚠️ {len(ans.unsupported)} sentence(s) may not be fully supported by the sources"):
            for s in ans.unsupported:
                st.write("• " + s)
    if ans.sources:
        with st.expander(f"📚 Sources ({len(ans.sources)})", expanded=ans.refused):
            for i, s in enumerate(ans.sources, 1):
                badge = " ✅ cited" if i in ans.cited else ""
                st.markdown(f"**[{i}] {s.source} — page {s.page}** · relevance {s.confidence:.0%}{badge}")
                st.caption(f"semantic {s.dense:.2f} · keyword coverage {s.coverage:.0%}" + (f" · re-rank {s.rerank:.2f}" if s.rerank is not None else ""))
                st.markdown(f"<div class='src-text'>{highlight_html(s.text, query)}</div>", unsafe_allow_html=True)
                st.divider()


tab_chat, tab_eval, tab_how = st.tabs(["💬 Chat", "📊 Evaluation", "🧠 How it works"])

# --------------------------------------------------------------------- chat
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not docs:
        st.info("👈 Upload a PDF in the sidebar, or click **Load sample** to try the demo HR policy.")
    elif not st.session_state.messages:
        st.write("Try one of these:")
        examples = ["How many casual leave days do I get?", "What is the hotel limit in metro cities?", "What are the core hours for work from home?", "What is the CEO's favourite colour?"]
        cols = st.columns(len(examples))
        for col, q in zip(cols, examples):
            col.button(q, on_click=lambda q=q: st.session_state.update(pending=q), use_container_width=True)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m.get("answer") and m["answer"].refused:
                st.warning(m["content"])
            else:
                st.markdown(m["content"])
            if m.get("answer"):
                render_details(m["answer"], m["query"])

    prompt = st.chat_input("Ask a question about your documents...") or st.session_state.pop("pending", None)
    if prompt:
        # refused questions are left out so a follow-up never latches onto an unanswered question
        history, skip_user = [], None
        msgs = st.session_state.messages
        for i, m in enumerate(msgs):
            if m["role"] == "user" and i + 1 < len(msgs) and msgs[i + 1].get("answer") is not None and msgs[i + 1]["answer"].refused:
                continue
            if m["role"] == "assistant" and m.get("answer") is not None and m["answer"].refused:
                continue
            history.append({"role": m["role"], "content": m["content"]})
        history = history[-6:]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if doc_filter == []:
                st.warning("Select at least one document in the sidebar.")
            else:
                with st.spinner("Searching your documents..."):
                    ans = pipe.ask(prompt, history=history, doc_ids=doc_filter, provider=provider,
                                   model=model or None, top_k=top_k, hybrid=hybrid, rerank=rerank, min_confidence=min_conf)
                if ans.refused:
                    st.warning(ans.answer)
                else:
                    st.markdown(ans.answer)
                render_details(ans, prompt)
                st.session_state.messages.append({"role": "assistant", "content": ans.answer, "answer": ans, "query": prompt})

# --------------------------------------------------------------- evaluation
with tab_eval:
    st.subheader("Measure retrieval quality")
    st.write(
        "Runs the questions in `eval/test_questions.json` against your **currently indexed documents** "
        "(designed for the sample Acme HR policy - click *Load sample* first). "
        "Unanswerable questions check that the guardrail correctly refuses."
    )
    k = st.slider("k (passages retrieved)", 1, 8, 4, key="eval_k")
    with_rerank = st.checkbox("Also test re-ranking (slower)")
    if st.button("Run evaluation", type="primary", disabled=not docs):
        with st.spinner("Evaluating..."):
            result = evaluate(pipe, load_questions(cfg.eval_file), k=k, include_rerank=with_rerank, min_confidence=min_conf)
        st.dataframe(pd.DataFrame(result["summary"]), use_container_width=True, hide_index=True)
        with st.expander("Per-question details"):
            st.dataframe(pd.DataFrame(result["details"]), use_container_width=True, hide_index=True)
        st.caption("Hit@k: correct passage in top-k · MRR: 1.0 means it is always ranked first · Refusal accuracy: unanswerable questions correctly refused.")

# ---------------------------------------------------------------------- how
with tab_how:
    st.markdown(
        """
### What happens when you ask a question
```
Question ──► (follow-up? rewrite using chat memory)
        ├─► Dense search   (embeddings, cosine similarity)  ─┐
        └─► Keyword search (BM25)                            ├─► Reciprocal Rank Fusion ─► optional re-rank
                                                             ┘
   ──► Confidence check ── low? ──► "I could not find this in the documents."
   ──► LLM (or extractive engine) writes an answer with [n] citations
   ──► Groundedness check: is every sentence supported by the sources?
```
**What makes this different from a basic "chat with PDF":** hybrid search, a refusal guardrail,
a groundedness check, follow-up rewriting, highlighted sources with page numbers, swappable LLMs
(including a no-LLM offline mode) and a built-in evaluation harness.
        """
    )
