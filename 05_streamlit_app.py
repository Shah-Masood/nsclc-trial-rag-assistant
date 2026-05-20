from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# -----------------------------
# Page config
# -----------------------------

st.set_page_config(
    page_title="NSCLC Clinical Trial RAG Assistant",
    page_icon="🧬",
    layout="wide",
)


# -----------------------------
# Paths / config
# -----------------------------

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

EMBEDDED_CHUNKS_PATH = PROCESSED_DIR / "embedded_chunks.parquet"
INDEX_PATH = PROCESSED_DIR / "faiss.index"

RAG_MODULE_PATHS = [
    Path("03_generateRAG.py"),
    Path("03_generateRag.py"),
    Path("03_generate_rag.py"),
]

DEFAULT_TOP_K = 8
DEFAULT_FETCH_K = 50
DEFAULT_MIN_SCORE = 0.75
DEFAULT_MAX_CONTEXT_CHARS = 16000
DEFAULT_MAX_TOKENS = 1600
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


# -----------------------------
# Utilities
# -----------------------------

def find_rag_module_path() -> Path:
    for path in RAG_MODULE_PATHS:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find your Phase 4 RAG file. Expected one of:\n"
        + "\n".join(str(p) for p in RAG_MODULE_PATHS)
    )


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def value_present(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, float) and pd.isna(value):
        return False

    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def source_url(row: pd.Series) -> Optional[str]:
    pmid = row.get("pmid")
    pmcid = row.get("pmcid")
    nct_id = row.get("nct_id")

    if value_present(pmid):
        return f"https://pubmed.ncbi.nlm.nih.gov/{str(pmid).strip()}/"

    if value_present(pmcid):
        pmcid_str = str(pmcid).strip()
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid_str}/"

    if value_present(nct_id):
        return f"https://clinicaltrials.gov/study/{str(nct_id).strip()}"

    return None


def make_source_label(row: pd.Series, source_num: int) -> str:
    parts = [f"S{source_num}"]

    nct_id = row.get("nct_id")
    pmid = row.get("pmid")
    pmcid = row.get("pmcid")
    source_type = row.get("source_type")
    section_name = row.get("section_name")

    if value_present(nct_id):
        parts.append(f"NCT:{nct_id}")

    if value_present(pmid):
        parts.append(f"PMID:{pmid}")

    if value_present(pmcid):
        parts.append(f"PMCID:{pmcid}")

    if value_present(source_type):
        parts.append(str(source_type))

    if value_present(section_name):
        parts.append(str(section_name))

    return "[" + " | ".join(parts) + "]"


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def linkify_answer(answer: str, retrieved_df: pd.DataFrame) -> str:
    """
    Converts exact source labels in the answer into clickable HTML links.

    Example:
    [S1 | NCT:NCT... | PMID:... | ...] → clickable PubMed/ClinicalTrials link
    """

    linked = html_escape(answer)

    for i, (_, row) in enumerate(retrieved_df.iterrows(), start=1):
        label = make_source_label(row, i)
        url = source_url(row)

        if not url:
            continue

        escaped_label = html_escape(label)

        link_html = (
            f'<a href="{url}" target="_blank" '
            f'style="text-decoration: none; font-weight: 600;">'
            f'{escaped_label}</a>'
        )

        linked = linked.replace(escaped_label, link_html)

    # Preserve markdown-ish line breaks.
    linked = linked.replace("\n", "<br>")

    return linked


def source_type_badge(source_type: str) -> str:
    if source_type == "pubmed_publication":
        return "📄 PubMed"
    if source_type == "clinicaltrials_registry":
        return "🧪 ClinicalTrials.gov"
    return "🔎 Source"


# -----------------------------
# Cached loading
# -----------------------------

@st.cache_resource(show_spinner=True)
def load_rag_module():
    rag_path = find_rag_module_path()
    return load_module_from_path("rag_app_module", rag_path)


@st.cache_resource(show_spinner=True)
def load_resources():
    st.write("Loading RAG module...")

    import faiss
    from sentence_transformers import SentenceTransformer

    rag_module = load_rag_module()

    st.write("Loading embedded chunks, FAISS index, and embedding model...")

    if hasattr(rag_module, "load_resources"):
        embed_model, index, chunks_df = rag_module.load_resources()
        return rag_module, embed_model, index, chunks_df

    raise AttributeError(
        "Your Phase 4 RAG module must expose a load_resources() function."
    )


# -----------------------------
# UI helpers
# -----------------------------

def render_header() -> None:
    st.title("🧬 NSCLC Clinical Trial RAG Assistant")
    st.caption(
        "Citation-grounded assistant for NSCLC immunotherapy and targeted therapy trials. "
        "Uses ClinicalTrials.gov registry chunks, linked PubMed publications, biomedical embeddings, FAISS retrieval, and Claude generation."
    )


def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("⚙️ Retrieval settings")

    top_k = st.sidebar.slider(
        "Top-K chunks sent to Claude",
        min_value=3,
        max_value=15,
        value=DEFAULT_TOP_K,
        step=1,
    )

    fetch_k = st.sidebar.slider(
        "FAISS fetch-K before filtering",
        min_value=10,
        max_value=100,
        value=DEFAULT_FETCH_K,
        step=5,
    )

    min_score = st.sidebar.slider(
        "Minimum similarity score",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_MIN_SCORE,
        step=0.01,
    )

    source_type_label = st.sidebar.selectbox(
        "Source filter",
        options=[
            "All sources",
            "ClinicalTrials.gov registry only",
            "PubMed publications only",
        ],
    )

    source_type = None
    if source_type_label == "ClinicalTrials.gov registry only":
        source_type = "clinicaltrials_registry"
    elif source_type_label == "PubMed publications only":
        source_type = "pubmed_publication"

    prefer_registry = st.sidebar.checkbox(
        "Boost registry/eligibility chunks",
        value=False,
    )

    prefer_publications = st.sidebar.checkbox(
        "Boost PubMed/result chunks",
        value=False,
    )

    nct_id = st.sidebar.text_input(
        "Optional NCT ID filter",
        placeholder="NCT03875950",
    ).strip()

    st.sidebar.divider()
    st.sidebar.header("🧠 Claude settings")

    claude_model = st.sidebar.text_input(
        "Claude model",
        value=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
    ).strip()

    max_context_chars = st.sidebar.slider(
        "Max context characters",
        min_value=4000,
        max_value=30000,
        value=DEFAULT_MAX_CONTEXT_CHARS,
        step=1000,
    )

    max_tokens = st.sidebar.slider(
        "Max output tokens",
        min_value=500,
        max_value=3000,
        value=DEFAULT_MAX_TOKENS,
        step=100,
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
    )

    force_generation = st.sidebar.checkbox(
        "Force generation even if retrieval is weak",
        value=False,
        help="Useful for debugging only. Leave off for serious demo.",
    )

    st.sidebar.divider()
    st.sidebar.header("🔐 API status")

    load_dotenv()
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))

    if has_key:
        st.sidebar.success("ANTHROPIC_API_KEY loaded")
    else:
        st.sidebar.error("Missing ANTHROPIC_API_KEY in .env")

    return {
        "top_k": top_k,
        "fetch_k": fetch_k,
        "min_score": min_score,
        "source_type": source_type,
        "prefer_registry": prefer_registry,
        "prefer_publications": prefer_publications,
        "nct_id": nct_id or None,
        "claude_model": claude_model,
        "max_context_chars": max_context_chars,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "force_generation": force_generation,
    }


def render_source_cards(retrieved_df: pd.DataFrame) -> None:
    st.subheader("Retrieved sources")

    if retrieved_df.empty:
        st.warning("No sources retrieved with the current settings.")
        return

    for i, (_, row) in enumerate(retrieved_df.iterrows(), start=1):
        label = make_source_label(row, i)
        url = source_url(row)
        source_type = clean_text(row.get("source_type"))
        section = clean_text(row.get("section_name"))
        title = clean_text(row.get("title"))
        score = float(row.get("similarity_score", 0.0))
        retrieval_score = float(row.get("retrieval_score", score))
        chunk_text = clean_text(row.get("chunk_text"))

        badge = source_type_badge(source_type)

        expander_title = (
            f"{i}. {badge} | score={score:.3f} | {section} | "
            f"{title[:90]}{'...' if len(title) > 90 else ''}"
        )

        with st.expander(expander_title, expanded=False):
            st.markdown(f"**Citation label:** `{label}`")

            if url:
                st.markdown(f"**Open source:** [Open link]({url})")

            meta_cols = st.columns(4)
            meta_cols[0].metric("Similarity", f"{score:.3f}")
            meta_cols[1].metric("Retrieval", f"{retrieval_score:.3f}")
            meta_cols[2].write(f"**Source:** {source_type}")
            meta_cols[3].write(f"**Section:** {section}")

            if value_present(row.get("nct_id")):
                st.write(f"**NCT ID:** `{row.get('nct_id')}`")

            if value_present(row.get("pmid")):
                st.write(f"**PMID:** `{row.get('pmid')}`")

            if value_present(row.get("pmcid")):
                st.write(f"**PMCID:** `{row.get('pmcid')}`")

            if title:
                st.write(f"**Title:** {title}")

            st.markdown("**Retrieved chunk:**")
            st.text_area(
                label=f"chunk_{i}",
                value=chunk_text,
                height=220,
                label_visibility="collapsed",
            )


def render_example_questions() -> None:
    examples = [
        "What biomarker testing is required for EGFR-targeted therapy enrollment?",
        "Summarize eligibility criteria for metastatic NSCLC trials.",
        "What are common eligibility criteria for KRAS G12C inhibitor trials in NSCLC?",
        "Compare primary endpoints across pembrolizumab plus chemotherapy trials.",
        "Which trials enroll patients with prior immunotherapy treatment?",
        "What evidence is available from published literature versus registry protocol records?",
    ]

    st.markdown("#### Example questions")

    cols = st.columns(2)

    for idx, q in enumerate(examples):
        with cols[idx % 2]:
            if st.button(q, key=f"example_{idx}", use_container_width=True):
                st.session_state["question"] = q


# -----------------------------
# Main app
# -----------------------------

def main() -> None:
    render_header()
    settings = render_sidebar()

    try:
        rag_module, embed_model, index, chunks_df = load_resources()
    except Exception as e:
        st.error("Failed to load RAG resources.")
        st.exception(e)
        st.stop()

    with st.expander("Dataset / index status", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Embedded chunks", f"{len(chunks_df):,}")
        col2.metric("FAISS vectors", f"{index.ntotal:,}")
        col3.metric("Unique NCT IDs", f"{chunks_df['nct_id'].nunique():,}")

        st.write(f"**Embedded chunks:** `{EMBEDDED_CHUNKS_PATH}`")
        st.write(f"**FAISS index:** `{INDEX_PATH}`")

    render_example_questions()

    st.divider()

    question = st.text_area(
        "Ask a clinical trial research question",
        value=st.session_state.get(
            "question",
            "What biomarker testing is required for EGFR-targeted therapy enrollment?",
        ),
        height=100,
        placeholder="Ask about eligibility, biomarkers, endpoints, interventions, comparators, or registry-vs-publication evidence...",
    )

    st.session_state["question"] = question

    run_button = st.button("Run RAG assistant", type="primary", use_container_width=True)

    if not run_button:
        st.info("Enter a question and click **Run RAG assistant**.")
        return

    if not question.strip():
        st.warning("Please enter a question.")
        return

    with st.spinner("Retrieving relevant chunks..."):
        try:
            retrieved_df = rag_module.retrieve(
                query=question,
                embed_model=embed_model,
                index=index,
                chunks_df=chunks_df,
                top_k=settings["top_k"],
                fetch_k=settings["fetch_k"],
                min_score=settings["min_score"],
                source_type=settings["source_type"],
                prefer_publications=settings["prefer_publications"],
                prefer_registry=settings["prefer_registry"],
                nct_id=settings["nct_id"],
            )
        except TypeError:
            # Fallback if your older retrieve() does not support nct_id.
            retrieved_df = rag_module.retrieve(
                query=question,
                embed_model=embed_model,
                index=index,
                chunks_df=chunks_df,
                top_k=settings["top_k"],
                fetch_k=settings["fetch_k"],
                min_score=settings["min_score"],
                source_type=settings["source_type"],
                prefer_publications=settings["prefer_publications"],
                prefer_registry=settings["prefer_registry"],
            )

            if settings["nct_id"]:
                retrieved_df = retrieved_df[
                    retrieved_df["nct_id"].astype(str).str.upper()
                    == settings["nct_id"].upper()
                ].copy()

    if retrieved_df.empty:
        st.warning(
            "No retrieved chunks passed the current filters/threshold. "
            "Try lowering the similarity threshold or removing source filters."
        )
        render_source_cards(retrieved_df)
        return

    with st.spinner("Generating citation-grounded answer with Claude..."):
        try:
            answer = rag_module.generate_answer(
                question=question,
                retrieved_df=retrieved_df,
                model_name=settings["claude_model"],
                max_context_chars=settings["max_context_chars"],
                max_tokens=settings["max_tokens"],
                temperature=settings["temperature"],
                force_generation=settings["force_generation"],
            )
        except Exception as e:
            st.error("Generation failed.")
            st.exception(e)
            render_source_cards(retrieved_df)
            return

    st.subheader("Answer")

    linked_answer = linkify_answer(answer, retrieved_df)
    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(128,128,128,0.25);
            background: rgba(128,128,128,0.06);
            line-height: 1.55;
        ">
            {linked_answer}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    render_source_cards(retrieved_df)

    st.divider()

    with st.expander("Raw answer text", expanded=False):
        st.text_area(
            "Raw answer",
            value=answer,
            height=300,
            label_visibility="collapsed",
        )

    with st.expander("Raw retrieved dataframe", expanded=False):
        display_cols = [
            c for c in [
                "nct_id",
                "pmid",
                "pmcid",
                "source_type",
                "section_name",
                "similarity_score",
                "retrieval_score",
                "title",
                "chunk_text",
            ]
            if c in retrieved_df.columns
        ]

        st.dataframe(
            retrieved_df[display_cols],
            use_container_width=True,
            height=400,
        )


if __name__ == "__main__":
    main()