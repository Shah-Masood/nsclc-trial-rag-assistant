from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Optional

import anthropic
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config
# -----------------------------

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

EMBEDDED_CHUNKS_PATH = PROCESSED_DIR / "embedded_chunks.parquet"
INDEX_PATH = PROCESSED_DIR / "faiss.index"

# Must match your Phase 3 embedding model.
EMBED_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

DEFAULT_TOP_K = 8
DEFAULT_FETCH_K = 50
MIN_SIMILARITY_SCORE = 0.75

MAX_CONTEXT_CHARS = 16000

# High-performance default. You can override from CLI.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 1600
DEFAULT_TEMPERATURE = 0.1

# -----------------------------
# Patient-advice pre-flight guard
# -----------------------------

PATIENT_ADVICE_PATTERNS = [
    re.compile(r"\bshould\s+(i|we|my|she|he|they)\b.{0,40}\b(enroll|take|join|try|use|start|receive|do)\b", re.IGNORECASE),
    re.compile(r"\bmy\s+(mother|father|mom|dad|husband|wife|son|daughter|sister|brother|patient|friend|relative|spouse|child|parent|grandmother|grandfather|aunt|uncle|cousin)\b", re.IGNORECASE),
    re.compile(r"\bis\s+(this|that|NCT\d{8})\s+(trial|study|treatment|drug|therapy)\s+(right|appropriate|good|suitable|safe)\s+for\b", re.IGNORECASE),
    re.compile(r"\bcan\s+(my|i|we)\b.{0,30}\benroll\b", re.IGNORECASE),
    re.compile(r"\benroll\s+(my|me|us)\b", re.IGNORECASE),
    re.compile(r"\b(i|my\s+\w+)\s+(have|has|had|was|were|am|is)\s+(diagnosed|been\s+diagnosed|just\s+diagnosed)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(treatment|drug|therapy|trial|option)\s+should\s+(i|we|my)\b", re.IGNORECASE),
    re.compile(r"\b(recommend|suggest)\s+(a|an|the\s+best)\s+(trial|study|treatment|drug|therapy)\s+for\s+(me|my|us)\b", re.IGNORECASE),
]

PATIENT_ADVICE_REFUSAL = (
    "I cannot provide patient-specific medical or enrollment advice. "
    "I can summarize trial eligibility criteria, study design, interventions, and "
    "reported outcomes from the retrieved sources, so you can discuss them with a "
    "treating oncologist. "
    "Please rephrase the question in a general form (e.g. 'What are the eligibility "
    "criteria for trial NCT04105153?' instead of 'Should my mother enroll in NCT04105153?')."
)


def is_patient_advice_question(question: str) -> bool:
    """Deterministic check: does this question ask for patient-specific advice?"""
    if not question:
        return False
    q = str(question)
    return any(p.search(q) for p in PATIENT_ADVICE_PATTERNS)

# -----------------------------
# Loading resources
# -----------------------------

def load_resources() -> tuple[SentenceTransformer, faiss.Index, pd.DataFrame]:
    if not EMBEDDED_CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EMBEDDED_CHUNKS_PATH}. Run Phase 3 first:\n"
            "python 02_embedIndex.py"
        )

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INDEX_PATH}. Run Phase 3 first:\n"
            "python 02_embedIndex.py"
        )

    print(f"Loading embedded chunks: {EMBEDDED_CHUNKS_PATH}")
    chunks_df = pd.read_parquet(EMBEDDED_CHUNKS_PATH)

    print(f"Loading FAISS index: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    if len(chunks_df) != index.ntotal:
        raise ValueError(
            f"Mismatch: embedded_chunks has {len(chunks_df)} rows, "
            f"but FAISS index has {index.ntotal} vectors."
        )

    if "chunk_id" in chunks_df.columns:
        expected = np.arange(len(chunks_df))
        actual = chunks_df["chunk_id"].to_numpy()

        if not np.array_equal(actual, expected):
            print(
                "Warning: chunk_id does not exactly match row order. "
                "FAISS lookup depends on row order, so verify this before trusting retrieval."
            )

    return embed_model, index, chunks_df


# -----------------------------
# Retrieval
# -----------------------------

def embed_query(query: str, embed_model: SentenceTransformer) -> np.ndarray:
    return embed_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")


def retrieve(
    
    query: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    min_score: float = MIN_SIMILARITY_SCORE,
    source_type: Optional[str] = None,
    prefer_publications: bool = False,
    prefer_registry: bool = False,
    nct_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    FAISS retrieval over Phase 3 embedded chunks.

    Steps:
    1. Embed the query using the same model used for indexing.
    2. Fetch more than needed from FAISS.
    3. Apply optional metadata filters.
    4. Apply light reranking boosts.
    5. Drop weak matches.
    6. Return top_k.
    """

    nct_in_query = re.search(r"NCT\d{8}", query, re.IGNORECASE)
    if nct_in_query and not nct_id:
        nct_id = nct_in_query.group(0).upper()

# NEW: if NCT is specified, bypass FAISS and grab all chunks for that NCT
    if nct_id:
        nct_mask = chunks_df["nct_id"].astype(str).str.upper() == nct_id.upper()
        nct_chunks = chunks_df[nct_mask].copy()

        if nct_chunks.empty:
            return pd.DataFrame()  # NCT genuinely not in corpus

        # Assign max similarity since we're confident about relevance
        nct_chunks["similarity_score"] = 1.0
        nct_chunks["retrieval_score"] = 1.0

        return nct_chunks.head(top_k).reset_index(drop=True)

    # Otherwise, normal FAISS path:
    q_emb = embed_query(query, embed_model)
    search_k = min(max(fetch_k, top_k), index.ntotal)
    scores, indices = index.search(q_emb, search_k)

    q_emb = embed_query(query, embed_model)

    search_k = min(max(fetch_k, top_k), index.ntotal)
    scores, indices = index.search(q_emb, search_k)

    valid_indices = [idx for idx in indices[0] if idx >= 0]
    valid_scores = scores[0][:len(valid_indices)]

    results = chunks_df.iloc[valid_indices].copy()
    results["similarity_score"] = valid_scores

    if source_type:
        results = results[results["source_type"] == source_type].copy()

    if nct_id:
        results = results[results["nct_id"].astype(str).str.upper() == nct_id.upper()].copy()

    results = results[results["similarity_score"] >= min_score].copy()

    if results.empty:
        return results

    results["retrieval_score"] = results["similarity_score"]

    if prefer_publications:
        results.loc[
            results["source_type"].eq("pubmed_publication"),
            "retrieval_score",
        ] += 0.06

        results.loc[
            results["section_name"].astype(str).str.contains("result|conclusion|outcome", case=False, na=False),
            "retrieval_score",
        ] += 0.03

    if prefer_registry:
        results.loc[
            results["source_type"].eq("clinicaltrials_registry"),
            "retrieval_score",
        ] += 0.05

        results.loc[
            results["section_name"].astype(str).str.contains(
                "eligibility|intervention|arm_group|primary_outcomes|secondary_outcomes",
                case=False,
                na=False,
            ),
            "retrieval_score",
        ] += 0.03

    results = results.sort_values("retrieval_score", ascending=False)

    return results.head(top_k).reset_index(drop=True)


# -----------------------------
# Formatting
# -----------------------------

def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def value_present(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, float) and pd.isna(value):
        return False

    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def make_source_label(row: pd.Series, source_num: int) -> str:
    """
    Creates labels Claude can cite exactly.

    Example:
    [S1 | NCT:NCT03875950 | PMID:12345678 | clinicaltrials_registry | eligibility_inclusion]
    """

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


def format_context(
    retrieved_df: pd.DataFrame,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Converts retrieved chunks into a source-labeled context block.
    Claude must cite these labels.
    """

    blocks = []
    used_chars = 0

    for i, (_, row) in enumerate(retrieved_df.iterrows(), start=1):
        source_label = make_source_label(row, i)

        text = clean_text(row.get("chunk_text"))
        title = clean_text(row.get("title"))

        metadata_lines = [
            f"Source label: {source_label}",
            f"Title: {title or 'Unknown'}",
            f"Source type: {clean_text(row.get('source_type'))}",
            f"Evidence type: {clean_text(row.get('evidence_type'))}",
            f"Study type: {clean_text(row.get('study_type'))}",
            f"Overall status: {clean_text(row.get('overall_status'))}",
            f"Section: {clean_text(row.get('section_name'))}",
            f"Similarity score: {float(row.get('similarity_score', 0.0)):.3f}",
        ]

        block = "\n".join(metadata_lines) + "\nText:\n" + text + "\n"

        if used_chars + len(block) > max_context_chars:
            break

        blocks.append(block)
        used_chars += len(block)

    return "\n---\n".join(blocks)


def format_retrieval_debug(retrieved_df: pd.DataFrame) -> str:
    if retrieved_df.empty:
        return "No retrieval results passed the current filters/threshold."

    lines = []

    for i, (_, row) in enumerate(retrieved_df.iterrows(), start=1):
        label = make_source_label(row, i)
        preview = clean_text(row.get("chunk_text"))[:260]

        lines.append(
            f"{i}. {label}\n"
            f"   similarity={float(row.get('similarity_score', 0.0)):.3f} "
            f"retrieval={float(row.get('retrieval_score', 0.0)):.3f}\n"
            f"   title={clean_text(row.get('title'))[:120]}\n"
            f"   preview={preview}..."
        )

    return "\n".join(lines)


# -----------------------------
# Prompting
# -----------------------------

SYSTEM_PROMPT = """
You are a careful AI research assistant for oncology clinical trial literature review.

You specialize in NSCLC immunotherapy and targeted therapy trials.

You must answer using ONLY the retrieved context provided by the user. You are not allowed to use outside knowledge, even if you know the answer.

Core rules:
1. Every factual claim about a clinical trial, eligibility criterion, biomarker, endpoint, intervention, comparator, study status, result, publication, adverse event, sample size, or date must cite a source label.
2. Use source labels exactly as given, such as [S1 | NCT:NCT12345678 | PMID:12345678 | clinicaltrials_registry | eligibility_inclusion].
3. Do not invent NCT IDs, PMIDs, trial names, endpoints, biomarkers, thresholds, results, dates, adverse events, or enrollment criteria.
4. Clearly distinguish registry/protocol information from published literature.
5. If sources conflict, say they conflict and cite both.
6. If the retrieved context is incomplete, weak, or irrelevant, say there is not enough information from the retrieved sources.
7. Do not provide medical advice, patient-specific treatment recommendations, diagnosis, or eligibility decisions.
8. Use cautious research language: "the retrieved registry text states...", "the retrieved publication abstract reports...", "the provided sources do not show..."
9. Prefer precision over confidence.
10. SCOPE CHECK: Before answering, verify the retrieved context is actually about the same disease, intervention class, or topic as the user's question. 
    If the question is about a topic clearly outside the retrieved context's domain (e.g. user asks about breast cancer but retrieval returns NSCLC 
    trials), refuse with: "The retrieved context is about [topic of context], not [topic of question]. I cannot answer this from the provided sources."

Required answer format:
Direct answer:
- Briefly answer the question.

Evidence:
- Bullet points with citations on every factual claim.

Registry vs publication distinction:
- State whether the evidence came mainly from ClinicalTrials.gov registry chunks, PubMed publication chunks, or both.

Limitations / uncertainty:
- Mention missing fields, lack of published results, weak retrieval, or conflicts when relevant.
""".strip()


def build_user_prompt(question: str, context: str) -> str:
    return f"""
User question:
{question}

Retrieved context:
{context}

Instructions:
Answer the user question using ONLY the retrieved context.

Citation rules:
- Cite every factual claim with source labels from the retrieved context.
- Do not cite sources that do not support the claim.
- Do not make broad oncology claims unless the retrieved sources support them.
- If the context does not answer the question, say so clearly.

Now produce the answer.
""".strip()


def has_enough_retrieval(
    retrieved_df: pd.DataFrame,
    min_sources: int = 2,
    min_top_score: float = MIN_SIMILARITY_SCORE,
) -> bool:
    if retrieved_df.empty:
        return False

    if len(retrieved_df) < min_sources:
        return False

    top_score = float(retrieved_df["similarity_score"].max())

    if top_score < min_top_score:
        return False

    return True


def refusal_answer(retrieved_df: pd.DataFrame) -> str:
    if retrieved_df.empty:
        return (
            "I don’t have enough information from the retrieved sources to answer that safely. "
            "No chunks passed the current retrieval threshold/filter, so I should not infer clinical trial details."
        )

    top_score = float(retrieved_df["similarity_score"].max())

    return (
        "I don’t have enough information from the retrieved sources to answer that safely. "
        f"The best retrieved similarity score was {top_score:.3f}, which is not strong enough "
        "to support a cited clinical research answer without risking hallucination."
    )


# -----------------------------
# Anthropic generation
# -----------------------------

def get_anthropic_client() -> anthropic.Anthropic:
    """
    Loads ANTHROPIC_API_KEY from .env.
    """

    load_dotenv(".env")
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing ANTHROPIC_API_KEY.\n\n"
            "Add this to .env:\n"
            "ANTHROPIC_API_KEY=your_anthropic_api_key_here"
        )

    return anthropic.Anthropic(api_key=api_key)


def extract_text_from_anthropic_message(message: Any) -> str:
    """
    Anthropic returns content blocks. This safely joins text blocks.
    """

    parts = []

    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    return "\n".join(parts).strip()


def generate_answer(
    question: str,
    retrieved_df: pd.DataFrame,
    model_name: str = DEFAULT_CLAUDE_MODEL,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    force_generation: bool = False,
) -> str:
    if is_patient_advice_question(question):
        return PATIENT_ADVICE_REFUSAL
    
    if not force_generation and not has_enough_retrieval(retrieved_df):
        return refusal_answer(retrieved_df)

    context = format_context(
        retrieved_df,
        max_context_chars=max_context_chars,
    )

    user_prompt = build_user_prompt(question, context)

    client = get_anthropic_client()

    message = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    return extract_text_from_anthropic_message(message)


# -----------------------------
# Main RAG function
# -----------------------------

def answer_question(
    question: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    fetch_k: int = DEFAULT_FETCH_K,
    min_score: float = MIN_SIMILARITY_SCORE,
    source_type: Optional[str] = None,
    prefer_publications: bool = False,
    prefer_registry: bool = False,
    nct_id: Optional[str] = None,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    show_context: bool = False,
    force_generation: bool = False,
) -> str:
    retrieved_df = retrieve(
        query=question,
        embed_model=embed_model,
        index=index,
        chunks_df=chunks_df,
        top_k=top_k,
        fetch_k=fetch_k,
        min_score=min_score,
        source_type=source_type,
        prefer_publications=prefer_publications,
        prefer_registry=prefer_registry,
        nct_id=nct_id,
    )

    if show_context:
        print("\n========== Retrieved context ==========")
        print(format_retrieval_debug(retrieved_df))
        print("======================================\n")

    return generate_answer(
        question=question,
        retrieved_df=retrieved_df,
        model_name=claude_model,
        max_context_chars=max_context_chars,
        max_tokens=max_tokens,
        temperature=temperature,
        force_generation=force_generation,
    )


# -----------------------------
# Interactive CLI
# -----------------------------

EXAMPLE_QUESTIONS = [
    "What biomarker testing is required for EGFR-targeted therapy enrollment?",
    "What are common eligibility criteria for KRAS G12C inhibitor trials in NSCLC?",
    "Compare primary endpoints across pembrolizumab plus chemotherapy trials.",
    "Which trials enroll patients with prior immunotherapy treatment?",
    "Summarize eligibility criteria for patients with metastatic NSCLC.",
    "What evidence is available from published literature versus registry protocol records?",
]


def print_examples() -> None:
    print("\nExample questions:")
    for q in EXAMPLE_QUESTIONS:
        print(f"- {q}")
    print()


def run_interactive(
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    print("\nPhase 4 Anthropic RAG assistant loaded.")
    print("Ask a question, or type 'examples', 'quit', or 'exit'.\n")

    while True:
        question = input("Question> ").strip()

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break

        if question.lower() == "examples":
            print_examples()
            continue

        answer = answer_question(
            question=question,
            embed_model=embed_model,
            index=index,
            chunks_df=chunks_df,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            min_score=args.min_score,
            source_type=args.source_type,
            prefer_publications=args.prefer_publications,
            prefer_registry=args.prefer_registry,
            nct_id=args.nct_id,
            claude_model=args.claude_model,
            max_context_chars=args.max_context_chars,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            show_context=args.show_context,
            force_generation=args.force_generation,
        )

        print("\n========== Answer ==========")
        print(answer)
        print("============================\n")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Ask one question and exit. If omitted, launches interactive mode.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved chunks sent to Claude.",
    )

    parser.add_argument(
        "--fetch_k",
        type=int,
        default=DEFAULT_FETCH_K,
        help="Number of FAISS hits fetched before filters/reranking.",
    )

    parser.add_argument(
        "--min_score",
        type=float,
        default=MIN_SIMILARITY_SCORE,
        help="Minimum similarity score required for a chunk.",
    )

    parser.add_argument(
        "--source_type",
        type=str,
        default=None,
        choices=["clinicaltrials_registry", "pubmed_publication"],
        help="Optional source filter.",
    )

    parser.add_argument(
        "--prefer_publications",
        action="store_true",
        help="Boost PubMed publication chunks and result/conclusion/outcome sections.",
    )

    parser.add_argument(
        "--prefer_registry",
        action="store_true",
        help="Boost registry chunks and eligibility/intervention/outcome sections.",
    )

    parser.add_argument(
        "--nct_id",
        type=str,
        default=None,
        help="Optional filter to one NCT ID, e.g. NCT03875950.",
    )

    parser.add_argument(
        "--claude_model",
        type=str,
        default=os.getenv("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL),
        help="Anthropic Claude model ID.",
    )

    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=MAX_CONTEXT_CHARS,
        help="Maximum retrieved context characters included in the prompt.",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum output tokens from Claude.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Generation temperature.",
    )

    parser.add_argument(
        "--show_context",
        action="store_true",
        help="Print retrieved chunks before Claude's answer.",
    )

    parser.add_argument(
        "--force_generation",
        action="store_true",
        help="Bypass weak-retrieval refusal gate. Use only for debugging.",
    )

    args = parser.parse_args()

    embed_model, index, chunks_df = load_resources()

    if args.question:
        answer = answer_question(
            question=args.question,
            embed_model=embed_model,
            index=index,
            chunks_df=chunks_df,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            min_score=args.min_score,
            source_type=args.source_type,
            prefer_publications=args.prefer_publications,
            prefer_registry=args.prefer_registry,
            nct_id=args.nct_id,
            claude_model=args.claude_model,
            max_context_chars=args.max_context_chars,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            show_context=args.show_context,
            force_generation=args.force_generation,
        )

        print("\n========== Answer ==========")
        print(answer)
        print("============================\n")

    else:
        run_interactive(embed_model, index, chunks_df, args)


if __name__ == "__main__":
    main()