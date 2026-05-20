from __future__ import annotations
 
import argparse
import importlib.util
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
 
import pandas as pd
from dotenv import load_dotenv
 
 
# -----------------------------
# Config
# -----------------------------
 
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
 
EVAL_SET_PATH = EVAL_DIR / "eval_set.jsonl"
 
# Phase 3 + 4 file paths. We load these modules dynamically because their
# filenames start with digits (not legal Python module names).
EMBED_MODULE_PATH = Path("02_embedIndex.py")
RAG_MODULE_PATH = Path("03_generateRag.py")
 
 
# -----------------------------
# Dynamic module loading
# -----------------------------
 
def load_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Phase 5 needs {path.name} in the working directory."
        )
 
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
 
 
# -----------------------------
# Eval set loading
# -----------------------------
 
REQUIRED_FIELDS = {
    "question_id",
    "question",
    "expected_nct_ids",
    "should_refuse",
    "question_type",
}
 
 
def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Eval set not found at {path}. "
            f"Copy eval_set_starter.jsonl to {path} and expand it."
        )
 
    items = []
 
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
 
            if not line or line.startswith("#"):
                continue
 
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Eval set line {line_no} is not valid JSON: {e}"
                )
 
            missing = REQUIRED_FIELDS - set(item.keys())
 
            if missing:
                raise ValueError(
                    f"Eval set line {line_no} ({item.get('question_id', '?')}) "
                    f"missing required fields: {sorted(missing)}"
                )
 
            items.append(item)
 
    return items
 
 
# -----------------------------
# Citation parsing
# -----------------------------
 
CITATION_PATTERN = re.compile(
    r"\[S(\d+(?:\s*,\s*S?\d+)*)\s*(?:\|([^\]]+))?\]",
    flags=re.IGNORECASE,
)
 
NCT_IN_LABEL_PATTERN = re.compile(r"NCT[:\s]*([A-Z0-9]+)", flags=re.IGNORECASE)
PMID_IN_LABEL_PATTERN = re.compile(r"PMID[:\s]*(\d+)", flags=re.IGNORECASE)
 
 
def parse_citations(answer_text: str) -> List[Dict[str, Optional[str]]]:
    """
    Extract every [S# | ... ] citation from the generated answer.
 
    Returns a list of dicts with:
        source_num: int           the S# index referenced
        nct_id:     Optional[str]
        pmid:       Optional[str]
        raw_label:  str           the entire matched label
    """
    citations = []
    for match in CITATION_PATTERN.finditer(answer_text or ""):
        nums_str = match.group(1)
        body = match.group(2) or ""
        nums = [int(n.strip().lstrip("S")) for n in nums_str.split(",") if n.strip()]
        nct_match = NCT_IN_LABEL_PATTERN.search(body)
        pmid_match = PMID_IN_LABEL_PATTERN.search(body)
        for n in nums:
            citations.append({
                "source_num": n,
                "nct_id": nct_match.group(1).upper() if nct_match else None,
                "pmid": pmid_match.group(1) if pmid_match else None,
                "raw_label": match.group(0),
            })
    return citations
 
 
# -----------------------------
# Refusal detection
# -----------------------------
 
REFUSAL_PHRASES = [
    "don't have enough information from the retrieved",
    "do not have enough information from the retrieved",
    "not enough information from the retrieved sources",
    "insufficient information from the retrieved sources",
    "cannot answer this based on the provided",
    "cannot provide patient-specific",
    "should not provide medical advice",
    "the retrieved context is about",
    "cannot provide medical advice",
    "cannot provide medical or patient",
    "cannot provide medical advice or patient-specific",
    "patient-specific enrollment recommendations",
]
 
 
def detect_refusal(answer_text: str) -> bool:
    if not answer_text:
        return True
 
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)
 
 
# -----------------------------
# Retrieval metrics
# -----------------------------
 
def normalize_nct(nct: Any) -> str:
    return str(nct or "").strip().upper()
 
 
def retrieved_nct_ranks(retrieved_df: pd.DataFrame) -> List[str]:
    """
    Returns a list of NCT IDs in rank order. May contain repeats if the same
    NCT appears in multiple chunks. De-dup is the caller's choice.
    """
    if retrieved_df is None or retrieved_df.empty:
        return []
 
    return [normalize_nct(v) for v in retrieved_df["nct_id"].tolist()]
 
 
def hit_rate_at_k(expected: List[str], ranked: List[str], k: int) -> Optional[float]:
    expected_set = {normalize_nct(e) for e in expected if e}
 
    if not expected_set:
        return None  # Can't compute without labels.
 
    top_k = set(ranked[:k])
    return 1.0 if expected_set & top_k else 0.0
 
 
def recall_at_k(expected: List[str], ranked: List[str], k: int) -> Optional[float]:
    expected_set = {normalize_nct(e) for e in expected if e}
 
    if not expected_set:
        return None
 
    top_k = set(ranked[:k])
    found = expected_set & top_k
 
    return len(found) / len(expected_set)
 
 
def mean_reciprocal_rank(expected: List[str], ranked: List[str]) -> Optional[float]:
    expected_set = {normalize_nct(e) for e in expected if e}
 
    if not expected_set:
        return None
 
    for i, nct in enumerate(ranked, start=1):
        if nct in expected_set:
            return 1.0 / i
 
    return 0.0
 
 
# -----------------------------
# Citation faithfulness
# -----------------------------
 
def citation_faithfulness(
    citations: List[Dict[str, Optional[str]]],
    retrieved_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Lexical faithfulness:
    A citation is "supported" if the NCT and/or PMID it names actually
    appears in the retrieved chunks (the chunks Claude was shown).
 
    A citation that names an NCT or PMID NOT in retrieval is hallucinated.
    """
 
    n_citations = len(citations)
    n_unique_sources = len({c["source_num"] for c in citations})
 
    if retrieved_df is None or retrieved_df.empty:
        retrieved_ncts = set()
        retrieved_pmids = set()
    else:
        retrieved_ncts = {normalize_nct(v) for v in retrieved_df["nct_id"].tolist() if v}
        retrieved_pmids = {
            str(v).strip() for v in retrieved_df["pmid"].tolist() if v and str(v) != "nan"
        }
 
    n_hallucinated_ncts = 0
    n_hallucinated_pmids = 0
    n_supported = 0
    hallucinated_examples = []
 
    for cit in citations:
        nct = cit.get("nct_id")
        pmid = cit.get("pmid")
 
        nct_ok = (nct is None) or (nct in retrieved_ncts)
        pmid_ok = (pmid is None) or (pmid in retrieved_pmids)
 
        if not nct_ok:
            n_hallucinated_ncts += 1
 
        if not pmid_ok:
            n_hallucinated_pmids += 1
 
        if nct_ok and pmid_ok:
            n_supported += 1
        else:
            hallucinated_examples.append(cit["raw_label"])
 
    lexical_faithfulness = (n_supported / n_citations) if n_citations else None
 
    return {
        "n_citations": n_citations,
        "n_unique_sources_cited": n_unique_sources,
        "n_supported": n_supported,
        "n_hallucinated_ncts": n_hallucinated_ncts,
        "n_hallucinated_pmids": n_hallucinated_pmids,
        "lexical_faithfulness": lexical_faithfulness,
        "hallucinated_examples": hallucinated_examples[:5],  # cap for readability
    }
 
 
# -----------------------------
# Per-question runner
# -----------------------------
 
def run_one_question(
    item: Dict[str, Any],
    rag_module,
    embed_model,
    index,
    chunks_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Runs retrieval + (optionally) generation for one eval question.
    Computes all metrics. Returns a result row.
    """
 
    question = item["question"]
    expected = [normalize_nct(n) for n in item.get("expected_nct_ids", []) if n]
    should_refuse = bool(item.get("should_refuse", False))
 
    start = time.time()
    error_message = None
 
    try:
        retrieved_df = rag_module.retrieve(
            query=question,
            embed_model=embed_model,
            index=index,
            chunks_df=chunks_df,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            min_score=args.min_score,
            prefer_publications=args.prefer_publications,
            prefer_registry=args.prefer_registry,
        )
    except Exception as e:
        retrieved_df = pd.DataFrame()
        error_message = f"retrieve failed: {e}"
        traceback.print_exc()
 
    ranked = retrieved_nct_ranks(retrieved_df)
 
    retrieval_metrics = {
        "hit_at_5": hit_rate_at_k(expected, ranked, 5),
        "hit_at_10": hit_rate_at_k(expected, ranked, 10),
        "recall_at_5": recall_at_k(expected, ranked, 5),
        "recall_at_10": recall_at_k(expected, ranked, 10),
        "mrr": mean_reciprocal_rank(expected, ranked),
        "n_retrieved": len(retrieved_df) if retrieved_df is not None else 0,
    }
 
    answer_text = ""
    system_refused = False
    answer_metrics: Dict[str, Any] = {}
    citations: List[Dict[str, Optional[str]]] = []
 
    if args.retrieval_only:
        answer_text = "[generation skipped: --retrieval_only]"
    else:
        try:
            # Check if the system would refuse without even calling the LLM.
            system_refused = not rag_module.has_enough_retrieval(retrieved_df)
 
            answer_text = rag_module.generate_answer(
                question=question,
                retrieved_df=retrieved_df,
                model_name=args.claude_model,
                max_context_chars=args.max_context_chars,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                force_generation=False,
            )
 
            citations = parse_citations(answer_text)
            answer_metrics = citation_faithfulness(citations, retrieved_df)
 
        except Exception as e:
            error_message = (error_message + " | " if error_message else "") + f"generate failed: {e}"
            traceback.print_exc()
 
    model_refused = detect_refusal(answer_text)
    refused = system_refused or model_refused
 
    # Refusal correctness:
    #   TP = should_refuse=True  AND  refused=True   (correct refusal)
    #   TN = should_refuse=False AND  refused=False  (correct answer)
    #   FP = should_refuse=False AND  refused=True   (incorrect refusal — user wanted an answer)
    #   FN = should_refuse=True  AND  refused=False  (incorrect answer — model should have refused)
    if should_refuse and refused:
        refusal_outcome = "TP_correct_refusal"
    elif (not should_refuse) and (not refused):
        refusal_outcome = "TN_correct_answer"
    elif (not should_refuse) and refused:
        refusal_outcome = "FP_incorrect_refusal"
    else:
        refusal_outcome = "FN_incorrect_answer"
 
    duration = time.time() - start
 
    return {
        "question_id": item["question_id"],
        "question": question,
        "question_type": item.get("question_type"),
        "expected_nct_ids": expected,
        "should_refuse": should_refuse,
        "retrieval_metrics": retrieval_metrics,
        "retrieved_nct_ids_ranked": ranked[: args.top_k],
        "retrieved_source_types": (
            retrieved_df["source_type"].tolist() if not retrieved_df.empty else []
        ),
        "answer_text": answer_text,
        "answer_length_words": len(answer_text.split()),
        "system_refused": system_refused,
        "model_refused": model_refused,
        "refused": refused,
        "refusal_outcome": refusal_outcome,
        "citations": citations,
        "citation_metrics": answer_metrics,
        "duration_seconds": round(duration, 2),
        "error": error_message,
    }
 
 
# -----------------------------
# Aggregation
# -----------------------------
 
def mean_excluding_none(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
 
    if not clean:
        return None
 
    return sum(clean) / len(clean)
 
 
def aggregate(results: List[Dict[str, Any]], config_name: str) -> Dict[str, Any]:
    n = len(results)
 
    retrieval = {
        "mean_hit_at_5": mean_excluding_none([r["retrieval_metrics"]["hit_at_5"] for r in results]),
        "mean_hit_at_10": mean_excluding_none([r["retrieval_metrics"]["hit_at_10"] for r in results]),
        "mean_recall_at_5": mean_excluding_none([r["retrieval_metrics"]["recall_at_5"] for r in results]),
        "mean_recall_at_10": mean_excluding_none([r["retrieval_metrics"]["recall_at_10"] for r in results]),
        "mean_mrr": mean_excluding_none([r["retrieval_metrics"]["mrr"] for r in results]),
        "n_questions_with_labels": sum(
            1 for r in results if r["retrieval_metrics"]["hit_at_5"] is not None
        ),
    }
 
    citation = {
        "mean_lexical_faithfulness": mean_excluding_none([
            r["citation_metrics"].get("lexical_faithfulness")
            for r in results if r["citation_metrics"]
        ]),
        "mean_n_citations_per_answer": mean_excluding_none([
            r["citation_metrics"].get("n_citations")
            for r in results if r["citation_metrics"]
        ]),
        "n_answers_with_hallucinated_ncts": sum(
            1 for r in results
            if r["citation_metrics"].get("n_hallucinated_ncts", 0) > 0
        ),
        "n_answers_with_hallucinated_pmids": sum(
            1 for r in results
            if r["citation_metrics"].get("n_hallucinated_pmids", 0) > 0
        ),
    }
 
    outcomes = Counter_dict([r["refusal_outcome"] for r in results])
 
    refusal = {
        "n_should_refuse": sum(1 for r in results if r["should_refuse"]),
        "n_should_answer": sum(1 for r in results if not r["should_refuse"]),
        "TP_correct_refusal": outcomes.get("TP_correct_refusal", 0),
        "TN_correct_answer": outcomes.get("TN_correct_answer", 0),
        "FP_incorrect_refusal": outcomes.get("FP_incorrect_refusal", 0),
        "FN_incorrect_answer": outcomes.get("FN_incorrect_answer", 0),
    }
 
    if n > 0:
        refusal["accuracy"] = (
            refusal["TP_correct_refusal"] + refusal["TN_correct_answer"]
        ) / n
    else:
        refusal["accuracy"] = None
 
    # Per question type
    by_type: Dict[str, Dict[str, Any]] = {}
 
    for r in results:
        qt = r.get("question_type", "unknown")
        bucket = by_type.setdefault(qt, {
            "n": 0,
            "hit_at_5": [],
            "lexical_faithfulness": [],
            "refusal_outcomes": [],
        })
        bucket["n"] += 1
        bucket["hit_at_5"].append(r["retrieval_metrics"]["hit_at_5"])
 
        if r["citation_metrics"]:
            bucket["lexical_faithfulness"].append(r["citation_metrics"].get("lexical_faithfulness"))
 
        bucket["refusal_outcomes"].append(r["refusal_outcome"])
 
    by_type_summary = {}
 
    for qt, bucket in by_type.items():
        by_type_summary[qt] = {
            "n": bucket["n"],
            "mean_hit_at_5": mean_excluding_none(bucket["hit_at_5"]),
            "mean_lexical_faithfulness": mean_excluding_none(bucket["lexical_faithfulness"]),
            "refusal_outcomes": Counter_dict(bucket["refusal_outcomes"]),
        }
 
    timing = {
        "total_seconds": round(sum(r["duration_seconds"] for r in results), 2),
        "mean_seconds_per_question": (
            round(sum(r["duration_seconds"] for r in results) / n, 2) if n else 0
        ),
    }
 
    return {
        "config_name": config_name,
        "n_questions": n,
        "retrieval": retrieval,
        "citation": citation,
        "refusal": refusal,
        "by_question_type": by_type_summary,
        "timing": timing,
        "n_errors": sum(1 for r in results if r.get("error")),
    }
 
 
def Counter_dict(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
 
    for x in items:
        out[x] = out.get(x, 0) + 1
 
    return out
 
 
# -----------------------------
# Reporting
# -----------------------------
 
def print_summary(summary: Dict[str, Any]) -> None:
    print("\n========== Eval summary ==========")
    print(f"Config:      {summary['config_name']}")
    print(f"Questions:   {summary['n_questions']}")
    print(f"Errors:      {summary['n_errors']}")
    print(f"Total time:  {summary['timing']['total_seconds']}s "
          f"({summary['timing']['mean_seconds_per_question']}s/question)")
 
    print("\n--- Retrieval ---")
    r = summary["retrieval"]
    print(f"  Hit@5:           {fmt(r['mean_hit_at_5'])}   (n_labeled={r['n_questions_with_labels']})")
    print(f"  Hit@10:          {fmt(r['mean_hit_at_10'])}")
    print(f"  Recall@5:        {fmt(r['mean_recall_at_5'])}")
    print(f"  Recall@10:       {fmt(r['mean_recall_at_10'])}")
    print(f"  MRR:             {fmt(r['mean_mrr'])}")
 
    print("\n--- Citations ---")
    c = summary["citation"]
    print(f"  Lexical faithfulness:           {fmt(c['mean_lexical_faithfulness'])}")
    print(f"  Mean citations per answer:      {fmt(c['mean_n_citations_per_answer'])}")
    print(f"  Answers with hallucinated NCTs: {c['n_answers_with_hallucinated_ncts']}")
    print(f"  Answers with hallucinated PMIDs:{c['n_answers_with_hallucinated_pmids']}")
 
    print("\n--- Refusal correctness ---")
    f = summary["refusal"]
    print(f"  Should refuse:    {f['n_should_refuse']}   Should answer: {f['n_should_answer']}")
    print(f"  TP (correct refusal):     {f['TP_correct_refusal']}")
    print(f"  TN (correct answer):      {f['TN_correct_answer']}")
    print(f"  FP (incorrect refusal):   {f['FP_incorrect_refusal']}   <- you should have answered")
    print(f"  FN (incorrect answer):    {f['FN_incorrect_answer']}   <- you should have refused (worst failure)")
    print(f"  Accuracy:                 {fmt(f['accuracy'])}")
 
    print("\n--- By question type ---")
    for qt, bucket in summary["by_question_type"].items():
        print(f"  {qt} (n={bucket['n']}):")
        print(f"      hit@5={fmt(bucket['mean_hit_at_5'])}  "
              f"faithfulness={fmt(bucket['mean_lexical_faithfulness'])}")
        print(f"      refusals={bucket['refusal_outcomes']}")
 
    print("==================================\n")
 
 
def fmt(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3f}"
 
 
# -----------------------------
# Main
# -----------------------------
 
def main() -> None:
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--config_name", type=str, default="default",
                        help="Tag for this run. Used in output filenames.")
    parser.add_argument("--eval_set", type=str, default=str(EVAL_SET_PATH),
                        help="Path to JSONL eval set.")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Limit to first N questions (for fast iteration).")
    parser.add_argument("--retrieval_only", action="store_true",
                        help="Skip generation. Fast and free.")
 
    # Retrieval knobs
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--fetch_k", type=int, default=50)
    parser.add_argument("--min_score", type=float, default=0.75)
    parser.add_argument("--prefer_publications", action="store_true")
    parser.add_argument("--prefer_registry", action="store_true")
 
    # Generation knobs
    parser.add_argument("--claude_model", type=str, default="claude-sonnet-4-5-20250929")
    parser.add_argument("--max_context_chars", type=int, default=16000)
    parser.add_argument("--max_tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.1)
 
    args = parser.parse_args()
 
    load_dotenv()
 
    # Load Phase 3 + 4 modules
    print(f"Loading module: {EMBED_MODULE_PATH}")
    embed_mod = load_module_from_path("embed_mod", EMBED_MODULE_PATH)
 
    print(f"Loading module: {RAG_MODULE_PATH}")
    rag_module = load_module_from_path("rag_mod", RAG_MODULE_PATH)
 
    # Reuse the loader from the RAG module
    print("Loading embed model, FAISS index, embedded chunks...")
    embed_model, index, chunks_df = rag_module.load_resources()
    print(f"  {len(chunks_df)} chunks, index size {index.ntotal}")
 
    # Load eval set
    print(f"\nLoading eval set: {args.eval_set}")
    eval_set = load_eval_set(Path(args.eval_set))
    print(f"  {len(eval_set)} questions loaded")
 
    if args.max_questions:
        eval_set = eval_set[: args.max_questions]
        print(f"  Limiting to first {len(eval_set)} questions.")
 
    # Run
    print(f"\n========== Running eval ({args.config_name}) ==========")
    print(f"Retrieval: top_k={args.top_k} fetch_k={args.fetch_k} min_score={args.min_score} "
          f"prefer_pub={args.prefer_publications} prefer_reg={args.prefer_registry}")
 
    if args.retrieval_only:
        print("Generation: SKIPPED (--retrieval_only)")
    else:
        print(f"Generation: {args.claude_model} (T={args.temperature}, max_tokens={args.max_tokens})")
 
    results = []
 
    for i, item in enumerate(eval_set, start=1):
        qid = item["question_id"]
        q = item["question"]
        print(f"\n[{i}/{len(eval_set)}] {qid}: {q[:90]}{'...' if len(q) > 90 else ''}")
 
        result = run_one_question(
            item=item,
            rag_module=rag_module,
            embed_model=embed_model,
            index=index,
            chunks_df=chunks_df,
            args=args,
        )
 
        results.append(result)
 
        rm = result["retrieval_metrics"]
        cm = result.get("citation_metrics", {}) or {}
        print(
            f"    hit@5={fmt(rm['hit_at_5'])}  "
            f"recall@5={fmt(rm['recall_at_5'])}  "
            f"mrr={fmt(rm['mrr'])}  "
            f"faith={fmt(cm.get('lexical_faithfulness'))}  "
            f"refused={result['refused']}  "
            f"outcome={result['refusal_outcome']}  "
            f"({result['duration_seconds']}s)"
        )
 
    # Aggregate
    summary = aggregate(results, args.config_name)
 
    # Save outputs
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = EVAL_DIR / f"results_{args.config_name}_{ts}.jsonl"
    summary_path = EVAL_DIR / f"summary_{args.config_name}_{ts}.json"
 
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
 
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
 
    print(f"\nSaved per-question results: {results_path}")
    print(f"Saved summary:               {summary_path}")
 
    print_summary(summary)
 
 
if __name__ == "__main__":
    main()