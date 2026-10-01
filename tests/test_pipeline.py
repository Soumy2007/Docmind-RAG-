import os
from pathlib import Path

import pytest

os.environ["EMBEDDING_BACKEND"] = "hashing"  # fast, offline, deterministic

from docmind.chunker import chunk_pages
from docmind.config import Settings
from docmind.guardrails import cited_numbers, grounded_label
from docmind.loader import load_pages
from docmind.memory import looks_like_follow_up, rewrite_question
from docmind.models import Page
from docmind.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_docs" / "Acme_HR_Policy.pdf"


@pytest.fixture()
def pipe(tmp_path):
    p = RAGPipeline(Settings(chroma_dir=tmp_path / "chroma"))
    p.ingest_file(SAMPLE)
    return p


def test_chunks_respect_size_and_keep_page_numbers():
    text = " ".join(f"This is sentence number {i} about leave." for i in range(60))
    chunks = chunk_pages([Page(text=text, page=3, source="x.pdf", doc_id="abc")], size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 60 for c in chunks)
    assert all(c.page == 3 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_pdf_loads_with_pages():
    pages = load_pages(SAMPLE, "doc1")
    assert len(pages) == 4 and pages[0].page == 1


def test_ingest_is_idempotent(pipe):
    assert pipe.ingest_file(SAMPLE)["status"] == "already indexed"


def test_answer_with_source_and_page(pipe):
    ans = pipe.ask("How many casual leave days per year?")
    assert not ans.refused
    assert "12 days" in ans.answer
    assert ans.sources[0].page == 1
    assert ans.cited


def test_unanswerable_question_is_refused(pipe):
    ans = pipe.ask("What is the CEO's favourite colour?")
    assert ans.refused


def test_document_filter(pipe):
    assert pipe.retrieve("casual leave", doc_ids=["does-not-exist"]) == []


def test_delete_document(pipe):
    doc_id = pipe.list_documents()[0]["doc_id"]
    pipe.remove_document(doc_id)
    assert pipe.list_documents() == []
    assert pipe.ask("casual leave?").refused


def test_follow_up_rewriting_without_llm():
    history = [{"role": "user", "content": "How many casual leave days?"}, {"role": "assistant", "content": "12"}]
    assert looks_like_follow_up("What about earned leave?")
    assert "casual leave" in rewrite_question("What about earned leave?", history)
    assert rewrite_question("What is the notice period?", history) == "What is the notice period?"


def test_citation_parsing_and_labels():
    assert cited_numbers("a [1] b [2, 3] c [1]") == [1, 2, 3]
    assert grounded_label(0.9) == "High" and grounded_label(0.1) == "Low" and grounded_label(None) == "n/a"


# ---------------------------------------------------------------- LLM paths (mocked, no network)
def test_llm_answer_is_cited_and_scored(pipe, monkeypatch):
    from docmind import llm

    monkeypatch.setattr(llm, "chat", lambda messages, provider, model=None, **kw: "You get 12 days of casual leave per year [1].")
    ans = pipe.ask("How many casual leave days per year?", provider="groq")
    assert ans.provider == "groq" and not ans.refused
    assert ans.cited == [1]
    assert ans.grounded_label in {"High", "Medium"}


def test_llm_failure_falls_back_to_extractive(pipe, monkeypatch):
    from docmind import llm

    def boom(*a, **k):
        raise llm.LLMError("network down")

    monkeypatch.setattr(llm, "chat", boom)
    ans = pipe.ask("How many casual leave days per year?", provider="openai")
    assert ans.provider == "extractive" and "network down" in ans.note and "12 days" in ans.answer


def test_hallucinated_sentence_is_flagged(pipe, monkeypatch):
    from docmind import llm

    fake = "You get 12 days of casual leave per year [1]. Employees also receive free helicopter rides to Mars every Tuesday."
    monkeypatch.setattr(llm, "chat", lambda *a, **k: fake)
    ans = pipe.ask("How many casual leave days per year?", provider="groq")
    assert any("helicopter" in s for s in ans.unsupported)


def test_openai_compatible_response_parsing(monkeypatch):
    from docmind import llm

    class R:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": [{"message": {"content": " hello [1] "}}]}

    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: R())
    assert llm.chat([{"role": "user", "content": "hi"}], "groq") == "hello [1]"


def test_missing_api_key_gives_clear_error(monkeypatch):
    from docmind import llm

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.chat([{"role": "user", "content": "hi"}], "gemini")
