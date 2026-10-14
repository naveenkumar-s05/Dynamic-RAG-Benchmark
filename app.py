"""Streamlit dashboard for the Dynamic RAG Document Intelligence & Benchmarking System.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
for _p in (str(PROJECT_ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import (  # noqa: E402
    BENCHMARK_DIR,
    REPORTS_DIR,
    SAMPLE_DOCS_DIR,
    SETTINGS,
    UPLOAD_DIR,
    ensure_directories,
    redacted_settings_summary,
)
from rag_benchmark.chunking import STRATEGIES, STRATEGY_LABELS  # noqa: E402
from rag_benchmark.chunking.base import ChunkingConfig  # noqa: E402
from rag_benchmark.evaluation import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
    best_strategy,
    comparison_dataframe,
    export_all,
    leaderboard_dataframe,
    load_dataset,
    passages_dataframe,
    per_query_dataframe,
)
from rag_benchmark.generation import AnswerGenerator, provider_from_settings  # noqa: E402
from rag_benchmark.indexing import (  # noqa: E402
    embedding_available,
    get_embedding_service,
)
from rag_benchmark.parsers import (  # noqa: E402
    ContentType,
    PARSER_ROUTES,
    choose_route,
    parse_document,
)
from rag_benchmark.pipeline import collect_chunks, retrieve  # noqa: E402
from rag_benchmark.utils.tokens import tokenizer_name  # noqa: E402
from rag_benchmark.utils.validation import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    ValidationError,
    safe_filename,
    validate_extension,
    validate_file_size,
)

ensure_directories()
st.set_page_config(page_title="Dynamic RAG Benchmark", layout="wide")


# --------------------------------------------------------------------- state
def _init_state() -> None:
    st.session_state.setdefault("documents", [])          # List[ParsedDocument]
    st.session_state.setdefault("doc_signatures", {})     # name -> signature
    st.session_state.setdefault("stores", {})             # signature -> (store, children, parents)
    st.session_state.setdefault("benchmark_report", None)


def _config_signature(config: ChunkingConfig, strategy: str, corpus_sig: str) -> str:
    payload = json.dumps(
        {"strategy": strategy, "config": config.to_dict(), "corpus": corpus_sig,
         "model": SETTINGS.embedding_model},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _corpus_signature() -> str:
    docs = st.session_state["documents"]
    key = "|".join(sorted(f"{d.source_id}:{d.document_version}" for d in docs))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _parse_and_store(path: Path, name: str) -> None:
    doc = parse_document(path, source_name=name, enable_ocr=True)
    existing = {d.source_id: i for i, d in enumerate(st.session_state["documents"])}
    if doc.source_id in existing:
        st.session_state["documents"][existing[doc.source_id]] = doc
    else:
        st.session_state["documents"].append(doc)
    st.session_state["stores"] = {}  # invalidate cached indexes


def reset_stores() -> None:
    st.session_state["stores"] = {}


# ---------------------------------------------------------------- components
def _get_store(strategy: str, config: ChunkingConfig):
    corpus_sig = _corpus_signature()
    key = _config_signature(config, strategy, corpus_sig)
    cached = st.session_state["stores"].get(key)
    if cached is not None:
        return cached
    embedder = get_embedding_service(SETTINGS.embedding_model)
    children, parents = collect_chunks(
        st.session_state["documents"], strategy, config, embedder
    )
    from rag_benchmark.indexing.vector_store import VectorStore

    store = VectorStore(
        dimension=embedder.dimension,
        embedding_model=SETTINGS.embedding_model,
        strategy=strategy,
        chunking_config=config,
        corpus_id=corpus_sig,
    )
    store.build_from_chunks(children, embedder.encode)
    st.session_state["stores"][key] = (store, children, parents)
    return store, children, parents


def _chunking_controls(prefix: str) -> ChunkingConfig:
    with st.expander("Chunking configuration", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        target = c1.slider("Target tokens", 100, 1200, 400, 50, key=f"{prefix}_target")
        hard_max = c2.slider("Hard max tokens", target, 2000, max(700, target), 50, key=f"{prefix}_max")
        overlap = c3.slider("Overlap tokens", 0, 300, 60, 10, key=f"{prefix}_overlap")
        min_tokens = c4.slider("Min tokens", 0, 300, 60, 10, key=f"{prefix}_min")
        c5, c6 = st.columns(2)
        threshold = c5.slider("Semantic threshold", 0.0, 1.0, 0.55, 0.05, key=f"{prefix}_thr")
        respect = c6.checkbox("Respect section boundaries", value=True, key=f"{prefix}_respect")
    config = ChunkingConfig(
        target_tokens=target,
        max_tokens=max(hard_max, target),
        overlap_tokens=min(overlap, target - 1 if target > 1 else 0),
        min_tokens=min(min_tokens, target),
        semantic_threshold=threshold,
        respect_sections=respect,
    )
    try:
        config.validate()
    except ValueError as exc:
        st.error(f"Invalid chunking configuration: {exc}")
        st.stop()
    return config


# -------------------------------------------------------------------- pages
def page_overview() -> None:
    st.title("Dynamic RAG Document Intelligence & Benchmarking System")
    st.markdown(
        """
        This application investigates how **dynamic parsing** and **chunking**
        choices affect retrieval quality. It is a fair-comparison benchmark first,
        and a grounded answer generator second.
        """
    )
    st.subheader("Pipeline")
    st.markdown(
        "1. **Parse** — route documents to format-specific parsers that preserve structure and provenance.\n"
        "2. **Chunk** — apply one of six strategies over the parsed blocks.\n"
        "3. **Embed & index** — sentence-transformers embeddings in a FAISS index.\n"
        "4. **Retrieve** — dense, BM25 or hybrid retrieval with metadata filters.\n"
        "5. **Generate** *(optional)* — grounded answers with citations.\n"
        "6. **Benchmark** — the same questions against every strategy, same settings."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Strategies", len(STRATEGIES))
    c2.metric("Documents loaded", len(st.session_state["documents"]))
    c3.metric("Embedding model", SETTINGS.embedding_model.split("/")[-1])

    st.subheader("Chunking strategies")
    st.table(pd.DataFrame({"key": list(STRATEGY_LABELS), "name": list(STRATEGY_LABELS.values())}))

    st.subheader("Corpus / index status")
    docs = st.session_state["documents"]
    if docs:
        rows = [
            {
                "source": d.source_name,
                "type": d.source_type,
                "blocks": d.block_count,
                "pages": d.page_count,
                "warnings": len(d.warnings),
                "parser": d.parser_name,
            }
            for d in docs
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No documents loaded. Go to **Document Ingestion** or load the sample corpus.")


def page_ingestion() -> None:
    st.title("Document Ingestion")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload documents",
            type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Ingest uploaded files"):
            for item in uploaded:
                try:
                    data = item.getvalue()
                    validate_file_size(len(data), SETTINGS.max_upload_bytes)
                    safe = safe_filename(item.name)
                    dest = UPLOAD_DIR / f"{hashlib.sha1(data).hexdigest()[:8]}_{safe}"
                    dest.write_bytes(data)
                    _parse_and_store(dest, item.name)
                    st.success(f"Ingested {item.name}")
                except ValidationError as exc:
                    st.error(f"{item.name}: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{item.name}: parsing failed — {exc}")
    with col2:
        st.markdown("**Sample corpus**")
        st.caption("Load the bundled synthetic benchmark documents.")
        if st.button("Load sample documents"):
            for path in sorted(SAMPLE_DOCS_DIR.glob("*")):
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    _parse_and_store(path, path.name)
            st.success(f"Loaded {len(st.session_state['documents'])} document(s).")
        if st.button("Clear all documents"):
            st.session_state["documents"] = []
            st.session_state["stores"] = {}
            st.success("Cleared.")

    docs = st.session_state["documents"]
    if not docs:
        st.info("No parsed documents yet.")
        return

    st.subheader("Parsed documents")
    names = [d.source_name for d in docs]
    selected = st.selectbox("Inspect a document", names)
    doc = next(d for d in docs if d.source_name == selected)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", doc.metadata.get("route", doc.parser_name))
    c2.metric("Blocks", doc.block_count)
    c3.metric("Pages", doc.page_count or "n/a")
    c4.metric("Warnings", len(doc.warnings))

    if doc.warnings:
        for warning in doc.warnings:
            st.warning(warning)

    block_rows = [
        {
            "block_id": b.block_id,
            "type": b.content_type,
            "page": b.page_number,
            "heading_level": b.heading_level,
            "section": b.section_path_str,
            "table_id": b.table_id,
            "row": b.row_number,
            "status": b.extraction_status,
            "text": (b.text[:400] + "…") if len(b.text) > 400 else b.text,
        }
        for b in doc.blocks
    ]
    st.dataframe(pd.DataFrame(block_rows), use_container_width=True, height=420)


def page_chunk_explorer() -> None:
    st.title("Chunk Explorer")
    if not st.session_state["documents"]:
        st.info("Load documents first (Document Ingestion).")
        return

    strategies = st.multiselect(
        "Compare strategies",
        options=STRATEGIES,
        default=["fixed", "structural", "hybrid"],
        format_func=lambda s: STRATEGY_LABELS[s],
    )
    config = _chunking_controls("explorer")
    if not strategies:
        return

    summary = []
    chunk_frames = {}
    for strategy in strategies:
        chunks, _ = collect_chunks(
            st.session_state["documents"], strategy, config,
            get_embedding_service(SETTINGS.embedding_model),
        )
        chunk_frames[strategy] = chunks
        tokens = [c.token_count for c in chunks]
        oversized = sum(1 for c in chunks if c.token_count > config.max_tokens)
        summary.append(
            {
                "strategy": STRATEGY_LABELS[strategy],
                "chunks": len(chunks),
                "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
                "min": min(tokens) if tokens else 0,
                "max": max(tokens) if tokens else 0,
                "oversized": oversized,
            }
        )
    st.subheader("Chunk counts")
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    metric_df = pd.DataFrame(summary).set_index("strategy")[["chunks", "avg_tokens", "max"]]
    st.bar_chart(metric_df)

    st.subheader("Inspect chunks")
    strategy = st.selectbox(
        "Strategy", strategies, format_func=lambda s: STRATEGY_LABELS[s]
    )
    chunks = chunk_frames[strategy]
    if not chunks:
        st.warning("This strategy produced no chunks.")
        return
    rows = [
        {
            "chunk_id": c.chunk_id,
            "source": c.source_name,
            "type": c.content_type,
            "tokens": c.token_count,
            "page": c.page_range,
            "section": c.section_path_str,
            "table_id": c.table_id,
            "reason": c.boundary_reason,
            "text": c.text,
        }
        for c in chunks
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=500)

    oversized = [c for c in chunks if c.token_count > config.max_tokens]
    if oversized:
        st.warning(f"{len(oversized)} chunk(s) exceed the hard maximum of {config.max_tokens} tokens.")
    else:
        st.success("No chunk exceeds the hard maximum.")


def page_retrieval() -> None:
    st.title("Retrieval Playground")
    if not st.session_state["documents"]:
        st.info("Load documents first (Document Ingestion).")
        return

    c1, c2, c3 = st.columns(3)
    strategy = c1.selectbox("Chunking strategy", STRATEGIES, format_func=lambda s: STRATEGY_LABELS[s])
    mode = c2.selectbox("Retrieval mode", ["dense", "bm25", "hybrid"])
    top_k = c3.slider("Top-k", 1, 15, 5)
    config = _chunking_controls("retrieval")

    with st.spinner("Building index…"):
        try:
            store, children, parents = _get_store(strategy, config)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to build index: {exc}")
            return

    st.caption(
        f"{store.size} chunks indexed · {SETTINGS.embedding_model} · tokenizer {tokenizer_name()}"
    )

    query = st.text_input("Question", "What encryption standard is used for data at rest?")
    cc1, cc2, cc3 = st.columns(3)
    expand_parents = cc1.checkbox("Expand to parent chunk", value=False)
    neigh = cc2.slider("Expand neighbours (window)", 0, 2, 0)
    gen = cc3.checkbox("Generate grounded answer", value=False)

    if not query:
        return

    embedder = get_embedding_service(SETTINGS.embedding_model)
    results = retrieve(
        store, query, mode=mode, top_k=top_k, embedder=embedder,
        expand_parents=expand_parents, parents=parents, expand_neighbors=neigh,
    )
    if not results:
        st.warning("No results. The index may be empty.")
        return

    for r in results:
        with st.expander(f"[{r.rank}] {r.source_name} · score {r.score:.3f} · {r.page_range or 'no page'} · {r.content_type}"):
            st.caption("Section: " + (" > ".join(r.section_path) or "n/a"))
            st.write(r.text)

    if gen:
        generator = AnswerGenerator(provider_from_settings(SETTINGS))
        answer = generator.generate(query, results)
        st.subheader("Grounded answer")
        if answer.error:
            st.info(answer.error)
        if answer.answer:
            st.write(answer.answer)
        if answer.citations:
            st.caption("Citations")
            st.dataframe(pd.DataFrame(answer.citations), use_container_width=True)
        if answer.insufficient_evidence:
            st.warning("The evidence was judged insufficient to answer.")


def page_benchmark() -> None:
    st.title("Benchmark Dashboard")
    st.markdown(
        "Runs the fixed question set against every selected strategy using the "
        "same parsed documents, embedding model, retrieval mode and top-k."
    )
    dataset_path = BENCHMARK_DIR / "questions.json"
    if not dataset_path.exists():
        st.error(f"Benchmark dataset not found at {dataset_path}.")
        return
    dataset = load_dataset(dataset_path)
    st.caption(
        f"Corpus **{dataset.corpus_id}** · {len(dataset.questions)} questions · "
        f"{len(dataset.documents)} documents"
    )

    c1, c2, c3 = st.columns(3)
    top_k = c1.slider("Top-k", 1, 10, 5)
    mode = c2.selectbox("Retrieval mode", ["dense", "bm25", "hybrid"], key="bench_mode")
    strategies = c3.multiselect(
        "Strategies", STRATEGIES, default=STRATEGIES, format_func=lambda s: STRATEGY_LABELS[s]
    )
    config = _chunking_controls("bench")

    if st.button("Run benchmark", type="primary"):
        with st.spinner("Running benchmark (embedding may take a moment)…"):
            bench_config = BenchmarkConfig(
                embedding_model=SETTINGS.embedding_model,
                strategies=strategies,
                chunking_config=config,
                top_k=top_k,
                retrieval_mode=mode,
                recall_ks=sorted({1, 3, top_k}),
                ndcg_k=top_k,
                persist_indexes=False,
            )
            runner = BenchmarkRunner(
                dataset, SAMPLE_DOCS_DIR, bench_config,
                embedder=get_embedding_service(SETTINGS.embedding_model),
            )
            report = runner.run_all(strategies)
            st.session_state["benchmark_report"] = report
            st.session_state["benchmark_paths"] = export_all(report, REPORTS_DIR)
        st.success("Benchmark complete.")

    report = st.session_state.get("benchmark_report")
    if report is None:
        st.info("Configure the run and click **Run benchmark**.")
        return

    comparison = comparison_dataframe(report)
    board = leaderboard_dataframe(report)

    st.subheader("Ranking")
    st.caption("Overall = mean of the three headline scores (all 0–1). Best first.")
    st.dataframe(board.round(3), use_container_width=True)

    metric_cols = [c for c in board.columns if c not in ("rank", "strategy", "overall")]
    if metric_cols:
        chart_df = board.set_index("strategy")[metric_cols].apply(pd.to_numeric, errors="coerce")
        st.bar_chart(chart_df)

    for metric in metric_cols:
        winner = best_strategy(report, metric)
        if winner:
            st.caption(f"Best **{metric}**: {winner[0]} ({winner[1]:.3f}). Winner may differ by metric.")

    with st.expander("Show all metrics"):
        st.dataframe(comparison, use_container_width=True)

    with st.expander("Per-query results"):
        st.dataframe(per_query_dataframe(report), use_container_width=True, height=360)

    with st.expander("Retrieved passages"):
        st.dataframe(passages_dataframe(report), use_container_width=True, height=360)

    st.subheader("Export")
    paths = st.session_state.get("benchmark_paths", {})
    for label, path in paths.items():
        if path.exists():
            st.download_button(
                f"Download {label}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/csv" if path.suffix == ".csv" else "application/json",
                key=f"dl_{label}",
            )
    if paths:
        st.caption("Saved under: " + str(REPORTS_DIR))


def page_diagnostics() -> None:
    st.title("Configuration & Diagnostics")

    st.subheader("Dependencies")
    import importlib

    deps = {
        "streamlit": "streamlit",
        "sentence-transformers": "sentence_transformers",
        "faiss": "faiss",
        "pymupdf": "fitz",
        "python-docx": "docx",
        "beautifulsoup4": "bs4",
        "pandas": "pandas",
        "rank-bm25": "rank_bm25",
        "tiktoken": "tiktoken",
        "openai": "openai",
        "pytesseract (optional)": "pytesseract",
    }
    rows = []
    for label, module in deps.items():
        try:
            importlib.import_module(module)
            rows.append({"dependency": label, "status": "available"})
        except Exception as exc:  # noqa: BLE001
            rows.append({"dependency": label, "status": f"missing ({exc.__class__.__name__})"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Embedding model")
    with st.spinner("Checking embedding model…"):
        ok, message = embedding_available(SETTINGS.embedding_model)
    (st.success if ok else st.error)(message)

    st.subheader("Generation provider")
    summary = redacted_settings_summary()
    provider = provider_from_settings(SETTINGS)
    st.write(
        {
            "configured_provider": summary["llm_provider"],
            "provider_ready": provider.is_configured(),
            "openai_api_key_configured": summary["openai_api_key_configured"],
            "openai_model": summary["openai_model"],
            "ollama_base_url": summary["ollama_base_url"],
            "ollama_model": summary["ollama_model"],
        }
    )
    if not provider.is_configured():
        st.info(
            "No LLM provider is active. Retrieval, chunking and benchmarking still "
            "work fully. Set LLM_PROVIDER plus credentials in `.env` to enable answers."
        )

    st.subheader("OCR")
    from rag_benchmark.parsers.ocr_parser import ocr_available

    st.write({"tesseract_available": ocr_available(), "tesseract_cmd_configured": summary["tesseract_cmd_configured"]})
    if not ocr_available():
        st.info(
            "OCR is unavailable. Install `pytesseract` and the Tesseract binary, "
            "then set TESSERACT_CMD if it is not on PATH. Scanned PDFs will be "
            "flagged as text-less until then."
        )

    st.subheader("Tokenization")
    st.write({"tokenizer": tokenizer_name()})


# --------------------------------------------------------------------- main
PAGES = {
    "Overview": page_overview,
    "Document Ingestion": page_ingestion,
    "Chunk Explorer": page_chunk_explorer,
    "Retrieval Playground": page_retrieval,
    "Benchmark Dashboard": page_benchmark,
    "Configuration & Diagnostics": page_diagnostics,
}


def main() -> None:
    _init_state()
    st.sidebar.title("Dynamic RAG Benchmark")
    choice = st.sidebar.radio("Navigation", list(PAGES))
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Embedding: `{SETTINGS.embedding_model.split('/')[-1]}`")
    st.sidebar.caption(f"Provider: `{SETTINGS.llm_provider}`")
    PAGES[choice]()


if __name__ == "__main__":
    main()
