# 📄 DocMind RAG - Document Q&A with hybrid search, guardrails and evaluation

Ask questions about your PDFs and get answers **with page-numbered citations**, a **confidence score**
and a **groundedness check**. Works fully offline (no API key) or with Ollama / Groq / Gemini / OpenAI / Claude.

## What makes it different from a basic "chat with PDF"

| Feature | Why it matters |
|---|---|
| **Hybrid search** (embeddings + BM25 keywords, fused with Reciprocal Rank Fusion) | Finds exact terms (`INR 6,000`, policy numbers) *and* paraphrases |
| **Optional cross-encoder re-ranking** | Higher precision on the final top-k |
| **Refusal guardrail** (blended confidence score) | Says *"I could not find this in the documents"* instead of guessing |
| **Groundedness check** | After answering, flags sentences the sources do not support (catches hallucinations) |
| **Sentence-aware chunking with page tracking** | No half-cut sentences; exact page citations |
| **Follow-up question rewriting** | "What about earned leave?" is understood using chat history |
| **Highlighted sources** | See exactly which passage produced the answer |
| **Swappable LLM, plus a no-LLM mode** | Runs for free, no key, no internet needed after first model download |
| **Built-in evaluation** (Hit@k, MRR, refusal accuracy, chunk-size sweep) | You can put real numbers on your resume |
| **Streamlit UI + FastAPI + CLI + Docker + tests** | Looks and behaves like a real project |

## Quick start (VS Code, Windows / Mac / Linux)

1. Install **Python 3.10 - 3.12** and open this folder in VS Code (`File > Open Folder`).
2. Open a terminal (`Ctrl + ~`) and create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows PowerShell / CMD
   source venv/bin/activate       # Mac / Linux
   ```
3. Install dependencies (first time takes a few minutes; `sentence-transformers` pulls PyTorch):
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```
5. In the browser: click **Load sample**, then ask *"How many casual leave days do I get?"*.
   Then upload your own PDF in the sidebar.

The first run downloads a ~90 MB embedding model (needs internet once). If it cannot download, the app
automatically switches to a small offline fallback and shows a warning - it still works, just less "semantic".

> In VS Code you can also press **F5** and pick *Streamlit app (debug)* (see `.vscode/launch.json`).

### Optional: use a real LLM for fluent answers
Copy `.env.example` to `.env` and set one provider, or just pick it in the sidebar:

| Provider | Setup |
|---|---|
| `ollama` (free, local) | Install Ollama, run `ollama pull llama3.2` |
| `groq` / `gemini` (free tiers) | Create an API key, put it in `.env` or the sidebar |
| `openai` / `anthropic` | Paid API key |

Model names change over time - if a provider returns "model not found", set `LLM_MODEL` in `.env`.
If an LLM call fails, the app falls back to the extractive engine and tells you why.

## Other ways to run

```bash
python cli.py ingest data/sample_docs     # index from terminal
python cli.py chat                        # chat in terminal
uvicorn api:app --reload                  # REST API, docs at http://127.0.0.1:8000/docs
python -m docmind.evaluation --sweep      # retrieval metrics + chunk-size comparison
pytest                                    # run tests
docker compose up --build             # see the Docker section below
```

## Run with Docker (optional)

1. Install **Docker Desktop** and start it (wait for "Engine running").
2. In the project folder run:
   ```bash
   docker compose up --build
   ```
   The first build takes several minutes (it downloads PyTorch). Later starts take seconds.
3. Open **http://localhost:8501**. Stop with `Ctrl + C`, or `docker compose down`.

Notes:
- Your indexed documents live in `./storage` and uploads in `./data/uploads`, so they survive restarts.
- API keys: copy `.env.example` to `.env` and fill it in; compose loads it automatically.
- Ollama on your laptop is reachable from the container at `http://host.docker.internal:11434` (already the default).
- REST API instead of the UI: `docker compose --profile api up --build` then open http://localhost:8000/docs.
- If port 8501 is busy, change `"8501:8501"` to e.g. `"8502:8501"` in `docker-compose.yml`.

## How it works

```
INDEXING   file -> load & clean (PyMuPDF) -> sentence-aware chunks (+page no.) -> embeddings -> ChromaDB
                                                                                      \-> BM25 keyword index
QUERYING   question -> (rewrite if follow-up)
           -> dense search + BM25 -> Reciprocal Rank Fusion -> (cross-encoder re-rank)
           -> confidence check ---- too low ----> "I could not find this in the documents."
           -> LLM / extractive answer with [n] citations
           -> groundedness check -> answer + sources + scores
```

## Project structure

```
docmind-rag/
├── app.py                 Streamlit UI (chat, evaluation tab, explainer)
├── api.py                 FastAPI backend (/upload, /ask, /documents)
├── cli.py                 Terminal interface
├── docmind/
│   ├── config.py          settings from .env
│   ├── loader.py          PDF/TXT/MD -> clean page text
│   ├── chunker.py         sentence-aware chunking with overlap
│   ├── embedder.py        sentence-transformers (+ offline fallback)
│   ├── vectorstore.py     ChromaDB wrapper
│   ├── retriever.py       hybrid search, RRF, re-ranking, confidence
│   ├── guardrails.py      confidence + groundedness scoring
│   ├── memory.py          follow-up question rewriting
│   ├── prompts.py         prompt templates
│   ├── llm.py             Ollama/Groq/Gemini/OpenAI/Claude + extractive engine
│   ├── pipeline.py        orchestrates everything
│   └── evaluation.py      Hit@k, MRR, refusal accuracy, chunk sweep
├── eval/test_questions.json
├── data/sample_docs/Acme_HR_Policy.pdf
├── scripts/make_sample_pdf.py
├── tests/test_pipeline.py
├── Dockerfile  requirements.txt  .env.example
```

## Tuning tips
- **Answers refused too often?** Lower *Refuse below confidence* in the sidebar. **Hallucinating?** Raise it.
  Use the **Evaluation** tab to see the effect.
- **Chunk size**: smaller = precise but less context; larger = more context but noisier. Use `--sweep` to compare.
- The sample PDF is tiny, so scores on it are easy. Add 5-10 of your own documents and write 30+ questions in
  `eval/test_questions.json` (fields: `question`, `evidence` = exact phrase from the document, `should_answer`).

## Resume bullets (fill in your own measured numbers)
- Built a RAG document Q&A system (Python, ChromaDB, Sentence-Transformers, BM25, FastAPI, Streamlit, Docker) with page-level citations.
- Implemented hybrid retrieval (dense + BM25 with Reciprocal Rank Fusion) and cross-encoder re-ranking; improved Hit@4 from __% to __% on a __-question test set.
- Added a confidence-based refusal guardrail and a groundedness checker that flags unsupported sentences, reducing hallucinated answers.
- Built an evaluation harness (Hit@k, MRR, refusal accuracy, chunk-size sweep) and unit tests; deployed the app with Docker.

## Interview questions this project prepares you for
Why chunk? How did you choose the chunk size? What is cosine similarity? Why hybrid search over pure vectors?
What is Reciprocal Rank Fusion? How do you reduce hallucination? How do you evaluate a RAG system? What happens
when the answer spans several chunks? How would you scale to 100,000 documents? (Answer: batch ingestion, a
server-based vector DB such as Qdrant/pgvector, async workers, caching.)

## Ideas to extend
OCR for scanned PDFs (Tesseract), streaming responses, table extraction, user login and per-user indexes,
DOCX/PPTX support, RAGAS metrics, and Hugging Face Spaces deployment.
