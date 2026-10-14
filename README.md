# Dynamic RAG Document Intelligence & Benchmarking System

A reproducible, end-to-end toolkit for **comparing document-chunking strategies in a
Retrieval-Augmented Generation (RAG) pipeline**. It ingests documents in several formats,
splits them with six different chunking strategies, indexes the chunks in a FAISS vector
store, runs a labeled question set against every strategy, and produces an apples-to-apples
comparison with retrieval-quality and efficiency metrics exported to CSV/JSON.

The retrieval and benchmarking pipeline runs **completely offline** with a local
sentence-transformers embedding model and FAISS — no API keys required. An optional
grounded-answer layer can call an OpenAI-compatible endpoint or a local Ollama server, but
it is disabled by default.

---

## Highlights

- **Six interchangeable chunking strategies** sharing one interface and configuration.
- **Multi-format ingestion**: PDF, DOCX, Markdown/TXT, HTML, CSV, and optional OCR for images.
- **Dense (FAISS), BM25, and hybrid (RRF) retrieval** behind a single factory.
- **Fair benchmarking**: identical documents, questions, embedding model, top-k, and
  retrieval mode across all strategies; each strategy gets its own index.
- **Rich metrics**: Recall@k, MRR, nDCG@k, context completeness, citation accuracy,
  duplicate-chunk rate, chunk-size statistics, and timing/size efficiency.
- **Streamlit dashboard** with six pages including a live Chunk Explorer, Retrieval
  Playground, and Benchmark Dashboard.
- **No fabrication**: in retrieval-only mode, generation-dependent metrics such as citation
  accuracy are reported as `null` (not applicable), never invented.
- **Deterministic tests**: the suite runs without downloading models or hitting the network.

---

## Project layout

```
dynamic-rag-benchmark/
├─ app.py                     # Streamlit UI (6 pages)
├─ config.py                  # Settings, directories, ensure_directories()
├─ conftest.py                # pytest path setup + shared fixtures (FakeEmbedder)
├─ requirements.txt
├─ .env.example               # copy to .env; all values optional
├─ data/
│  ├─ sample_documents/       # 6 sample docs (md, txt, html, csv, docx, pdf)
│  └─ benchmark/questions.json# 22 hand-authored, labeled questions
├─ scripts/
│  ├─ build_sample_docs.py    # (re)generate the DOCX/PDF samples
│  └─ run_benchmark.py        # CLI benchmark runner
├─ src/rag_benchmark/
│  ├─ parsers/                # router + per-format parsers -> ParsedDocument
│  ├─ chunking/               # base + 6 strategies + token splitting helpers
│  ├─ indexing/               # embeddings service + FAISS vector store
│  ├─ retrieval/              # dense, BM25, hybrid (RRF) retrievers
│  ├─ evaluation/            # dataset, metrics, benchmark runner, reporting
│  ├─ generation/             # optional grounded answering + citations
│  ├─ pipeline.py             # parse -> collect chunks -> build index -> retrieve
│  └─ utils/                  # tokens, logging, validation
├─ indexes/                   # generated FAISS indexes (git-ignored)
└─ reports/                   # generated CSV/JSON reports (git-ignored)
```

---

## The six chunking strategies

| Key | Label | Idea |
| --- | --- | --- |
| `fixed` | A. Fixed-size | Sliding window of N tokens with overlap. Simple, uniform, boundary-agnostic baseline. |
| `sentence` | B. Sentence-based | Groups whole sentences up to a target token budget; respects sentence boundaries. |
| `structural` | C. Structural | Splits on document structure (headings, paragraphs, lists, tables, code); keeps table headers with rows and code intact; also emits section parent chunks. |
| `recursive` | D. Recursive | Recursively splits on decreasing separators (section → paragraph → sentence) until chunks fit. |
| `semantic` | E. Semantic | Embeds sentence units and cuts where consecutive-sentence similarity drops below a threshold; falls back gracefully if no embedder is available. |
| `hybrid` | F. Hybrid / dynamic | Chooses structural boundaries but adapts to content and size, merging/refining dynamically. |

All strategies accept the same `ChunkingConfig(target_tokens, max_tokens, overlap_tokens,
min_tokens, semantic_threshold)` and emit `Chunk` objects with full provenance
(source, page range, section path, block ids, table id, parent chunk id, boundary reason).

---

## Metrics

Per question, per strategy:

- **Recall@k** — fraction of relevant chunks found in the top-k.
- **MRR** — reciprocal rank of the first relevant chunk.
- **nDCG@k** — graded ranking quality (grade 2 = primary fact, 1 = secondary fact, 0 = none).
- **Context completeness** — fraction of the question's essential facts present in retrieved text.
- **Citation accuracy** — precision/coverage of cited sources vs. expected evidence
  (*only when grounded answering is enabled; otherwise `null` → N/A*).

Aggregated per strategy:

- Mean metrics, **duplicate-chunk rate**, **chunk-size statistics** (avg/min/max/median),
  **oversized-chunk rate**, and **timing** (chunking, embedding, indexing, retrieval, total)
  plus **index size**.

Grading is deterministic and comes from hand-written labels in
`data/benchmark/questions.json` — relevance is **never** derived from retrieval output.

---

## Installation

Tested with **Python 3.11** on Windows (CPU only) and Linux.

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate          # Linux/macOS

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. (Optional) create your environment file
Copy-Item .env.example .env          # Windows
# cp .env.example .env               # Linux/macOS
```

The default embedding model `sentence-transformers/all-MiniLM-L6-v2` (~90 MB, 384 dims)
downloads on first use and is then cached. To run fully offline after the first download,
set `HF_HUB_OFFLINE=1`.

---

## Quick start

```powershell
# (Optional) regenerate the DOCX and PDF sample documents
python scripts\build_sample_docs.py

# Run the full benchmark over all six strategies and write reports to reports\
python scripts\run_benchmark.py

# Choose strategies / retrieval mode / top-k
python scripts\run_benchmark.py --strategies fixed structural hybrid --top-k 5
python scripts\run_benchmark.py --retrieval-mode hybrid --no-index-persist

# Launch the interactive dashboard
streamlit run app.py
```

`run_benchmark.py` prints a comparison table and writes four artifacts to `reports/`:
`*_comparison.csv`, `*_per_query.csv`, `*_retrieved_passages.csv`, and `*_report.json`.

### Streamlit pages

1. **Overview** — pipeline explanation and status.
2. **Document Ingestion** — upload/parse files; inspect extracted blocks and warnings.
3. **Chunk Explorer** — compare how each strategy chunks the same document side by side.
4. **Retrieval Playground** — query an index and inspect ranked passages with metadata.
5. **Benchmark Dashboard** — run the labeled benchmark and view/export comparisons.
6. **Configuration & Diagnostics** — embedding model, retrieval defaults, provider status
   (secrets are never displayed).

---

## Example result

The console prints a **ranked leaderboard**. `overall` is the mean of the three headline
scores (Recall@k, MRR, context completeness), all on a 0–1 scale, so the best strategy is
row 1. From the bundled `northwind-v1` corpus (6 documents, 22 questions, dense retrieval,
top-k=5):

| Rank | Strategy | Overall | Recall@5 | MRR | Context completeness |
| --- | --- | --- | --- | --- | --- |
| 1 | B. Sentence-based | 0.955 | 0.977 | 0.911 | 0.977 |
| 2 | C. Structural | 0.955 | 0.977 | 0.911 | 0.977 |
| 3 | E. Semantic | 0.955 | 0.977 | 0.911 | 0.977 |
| 4 | A. Fixed-size | 0.927 | 1.000 | 0.782 | 1.000 |
| 5 | D. Recursive | 0.924 | 1.000 | 0.773 | 1.000 |
| 6 | F. Hybrid / dynamic | 0.915 | 0.932 | 0.860 | 0.955 |

The full comparison (recall@1/@3, nDCG@5, chunk-size and timing stats, index size) is still
written to the CSV/JSON reports and is available under the dashboard's **Show all metrics**
expander. *(Numbers are illustrative of the bundled corpus and will vary with model,
documents and configuration. Citation accuracy is `null` here because generation is
disabled.)*

---

## Configuration

All settings come from environment variables (see `.env.example`) and have safe defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any sentence-transformers model id. |
| `HF_HUB_OFFLINE` | `0` | Set to `1` to prevent network calls once cached. |
| `DEFAULT_TOP_K` | `5` | Default number of retrieved chunks. |
| `DEFAULT_RETRIEVAL_MODE` | `dense` | `dense`, `bm25`, or `hybrid`. |
| `LLM_PROVIDER` | `none` | `none`, `openai`, or `ollama`. |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | — | OpenAI-compatible generation. |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3.2` | Local generation. |
| `TESSERACT_CMD` | — | Path to the Tesseract binary for OCR. |
| `MAX_UPLOAD_MB` | `25` | Max accepted upload size. |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

**Never commit a real `.env` or API key.** `.env` is git-ignored; `.env.example` is the
template.

---

## Testing

```powershell
python -m pytest -q
```

The suite uses a deterministic `FakeEmbedder` (no model download, no network) and covers
parsers, all six chunkers, metadata provenance, the embedding service, dense/BM25/hybrid
retrieval, metric math (hand-calculated), the end-to-end benchmark engine, report export,
and grounded-answer citation handling.

To also exercise the **real** embedding model (downloads/loads it), opt in explicitly:

```powershell
$env:RUN_EMBEDDING_TESTS="1"; python -m pytest -q
```

---

## Reproducibility & fairness notes

- Every strategy is evaluated on the **same** documents, the **same** questions, the
  **same** embedding model, the **same** `top_k`, and the **same** retrieval mode.
- Chunking is the only variable, so metric differences reflect chunking quality.
- Relevance labels are hand-written; grading is deterministic and independent of retrieval.
- Running the same command twice yields identical rankings and scores.

---

## Limitations

- **OCR is optional and off by default.** The OCR path requires the Tesseract binary; if it
  is not installed, image/scanned documents are skipped with a clear warning rather than
  silently ignored.
- **Citation accuracy is N/A in retrieval-only mode.** It is reported as `null`, not guessed.
- **LLM generation is optional.** When `LLM_PROVIDER=none`, the system returns retrieved
  evidence and citations but no synthesized answer.
- CPU-only execution is supported but slower for large corpora; GPU is used automatically
  if available.
- The bundled corpus is small; treat the sample table as a demonstration, not a benchmark
  of absolute quality.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Model download is slow or blocked | Pre-download once, then set `HF_HUB_OFFLINE=1`. |
| `faiss` import warning about AVX2 | Harmless; FAISS falls back to a non-AVX2 build. |
| OCR documents skipped | Install Tesseract and set `TESSERACT_CMD`. |
| Streamlit can't find modules | Run from the project root; `app.py` adds `src/` to `sys.path`. |
| `overlap_tokens must be smaller than target_tokens` | Lower `overlap_tokens` below `target_tokens`. |

---

## License

Provided for educational and internal evaluation use. Add a license before distributing.
