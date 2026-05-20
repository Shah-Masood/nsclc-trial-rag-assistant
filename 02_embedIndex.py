from __future__ import annotations
 
import argparse
import time
from pathlib import Path
from typing import List, Optional
 
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
 
 
# -----------------------------
# Config
# -----------------------------
 
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
 
CHUNKS_PATH = PROCESSED_DIR / "chunks.parquet"
EMBEDDED_CHUNKS_PATH = PROCESSED_DIR / "embedded_chunks.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
INDEX_PATH = PROCESSED_DIR / "faiss.index"
 
EMBED_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBED_BATCH_SIZE = 32
 
MAX_CHARS = 1000
OVERLAP_CHARS = 200
SENTENCE_BOUNDARY_LOOKBACK = 150  # chars searched backward for a sentence end
 
 
# -----------------------------
# Sub-chunking
# -----------------------------
 
def sub_chunk(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> List[str]:
    """
    Split a long string into overlapping windows, preferring sentence boundaries.
 
    Algorithm:
    1. If text fits in max_chars, return as-is.
    2. Walk through in (max_chars - overlap_chars) strides.
    3. For each window, look back up to SENTENCE_BOUNDARY_LOOKBACK chars from
       the tentative end for sentence-ending punctuation (".", "!", "?")
       followed by whitespace; if found, cut there.
    4. Next window starts at (end - overlap_chars) for continuity.
    """
    text = (text or "").strip()
 
    if not text:
        return []
 
    if len(text) <= max_chars:
        return [text]
 
    chunks: List[str] = []
    start = 0
    n = len(text)
 
    while start < n:
        end = min(start + max_chars, n)
 
        # Try to break at a sentence boundary near the window end.
        if end < n:
            search_floor = max(end - SENTENCE_BOUNDARY_LOOKBACK, start + 1)
 
            for idx in range(end - 1, search_floor - 1, -1):
                if text[idx] in ".!?":
                    next_idx = idx + 1
                    # Real sentence ends are followed by whitespace or EOF.
                    if next_idx >= n or text[next_idx] in " \n\t":
                        end = next_idx
                        break
 
        chunk = text[start:end].strip()
 
        if chunk:
            chunks.append(chunk)
 
        if end >= n:
            break
 
        new_start = end - overlap_chars
 
        # Guard against pathological loops.
        if new_start <= start:
            new_start = end
 
        start = new_start
 
    return chunks
 
 
def expand_chunks(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sub_chunk() to every row, preserving all metadata columns.
 
    New columns added:
        parent_chunk_idx   index in input chunks_df
        sub_chunk_idx      0-based position within the parent chunk
        sub_chunk_count    total sub-chunks created from the parent
        chunk_id           row index in the output frame, == FAISS row id
 
    chunk_text and chunk_char_length are overwritten with the sub-chunk values.
    """
    rows = []
 
    for parent_idx, row in tqdm(
        chunks_df.iterrows(),
        total=len(chunks_df),
        desc="Sub-chunking",
    ):
        sub_texts = sub_chunk(row["chunk_text"])
 
        for i, sub_text in enumerate(sub_texts):
            new_row = row.to_dict()
            new_row["chunk_text"] = sub_text
            new_row["chunk_char_length"] = len(sub_text)
            new_row["parent_chunk_idx"] = int(parent_idx)
            new_row["sub_chunk_idx"] = i
            new_row["sub_chunk_count"] = len(sub_texts)
            rows.append(new_row)
 
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["chunk_id"] = out.index  # == FAISS row id
    return out
 
 
# -----------------------------
# Embedding
# -----------------------------
 
def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """
    Encode texts to L2-normalized embeddings (so cosine sim == inner product).
    Returns float32 array of shape [len(texts), dim].
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
 
    return embeddings.astype("float32")
 
 
# -----------------------------
# FAISS index
# -----------------------------
 
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    IndexFlatIP over L2-normalized vectors == exact cosine similarity.
    For corpora up to ~100k vectors this is fast enough on a laptop.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index
 
 
# -----------------------------
# Search (Phase 4 will import this)
# -----------------------------
 
def search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    sub_chunks_df: pd.DataFrame,
    k: int = 5,
    filter_source_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Embed the query, run top-k inner-product search, join back to chunk metadata.
 
    Args:
        query: natural language question.
        model: the SentenceTransformer used to build the index. MUST match.
        index: the FAISS IndexFlatIP.
        sub_chunks_df: the parquet loaded from EMBEDDED_CHUNKS_PATH. Row order
            must equal the FAISS row order (use chunk_id column to verify).
        k: number of results.
        filter_source_type: optional. If set (e.g., "pubmed_publication"),
            results are filtered AFTER retrieval. For real metadata filtering
            at scale, use faiss.IDSelectorArray or move to a store like
            Chroma/Weaviate.
 
    Returns:
        DataFrame of top-k rows with a similarity_score column.
    """
    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
 
    # Over-fetch when filtering so we still return k results after the filter.
    search_k = k * 5 if filter_source_type else k
 
    scores, indices = index.search(q_emb, search_k)
 
    results = sub_chunks_df.iloc[indices[0]].copy()
    results["similarity_score"] = scores[0]
 
    if filter_source_type:
        results = results[results["source_type"] == filter_source_type]
 
    return results.head(k).reset_index(drop=True)
 
 
# -----------------------------
# Sanity & demo
# -----------------------------
 
def print_sanity_stats(
    chunks_df: pd.DataFrame,
    sub_chunks_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:
    print("\n========== Phase 3 sanity stats ==========")
    print(f"Original chunks:        {len(chunks_df)}")
    print(
        f"After sub-chunking:     {len(sub_chunks_df)}  "
        f"(inflation x{len(sub_chunks_df) / max(len(chunks_df), 1):.2f})"
    )
    print(f"Embedding dim:          {embeddings.shape[1]}")
    print(f"Embeddings shape:       {embeddings.shape}")
 
    print("\nSub-chunk char length distribution:")
    print(sub_chunks_df["chunk_char_length"].describe().to_string())
 
    print("\nSub-chunks per source_type:")
    print(sub_chunks_df["source_type"].value_counts().to_string())
 
    print("\nTop 10 section_name counts:")
    print(sub_chunks_df["section_name"].value_counts().head(10).to_string())
 
    print("\nTrials with chunks remaining:")
    if "nct_id" in sub_chunks_df.columns:
        print(f"  {sub_chunks_df['nct_id'].nunique()} unique NCT IDs")
 
 
DEMO_QUERIES = [
    "What are common eligibility criteria for KRAS G12C inhibitor trials in NSCLC?",
    "Compare primary endpoints across pembrolizumab plus chemotherapy trials.",
    "Which trials enroll patients with prior immunotherapy treatment?",
    "What biomarker testing is required for EGFR-targeted therapy enrollment?",
]
 
 
def run_demo_queries(
    model: SentenceTransformer,
    index: faiss.Index,
    sub_chunks_df: pd.DataFrame,
    k: int = 5,
) -> None:
    print("\n========== Demo retrieval (k=5) ==========")
 
    for query in DEMO_QUERIES:
        print(f"\n--- {query!r} ---")
        results = search(query, model, index, sub_chunks_df, k=k)
 
        for i, row in results.iterrows():
            print(
                f"  [{i + 1}] score={row['similarity_score']:.3f}  "
                f"nct={row.get('nct_id')}  "
                f"section={row.get('section_name')}  "
                f"len={row['chunk_char_length']}"
            )
            preview = (row["chunk_text"] or "")[:180].replace("\n", " ")
            print(f"      {preview}...")
 
 
# -----------------------------
# Main
# -----------------------------
 
def main() -> None:
    parser = argparse.ArgumentParser()
 
    parser.add_argument(
        "--demo_only",
        action="store_true",
        help="Skip embedding. Load existing index + parquet and run demo queries.",
    )
 
    args = parser.parse_args()
 
    if args.demo_only:
        print("Loading existing index + embedded chunks...")
        sub_chunks_df = pd.read_parquet(EMBEDDED_CHUNKS_PATH)
        index = faiss.read_index(str(INDEX_PATH))
        model = SentenceTransformer(EMBED_MODEL_NAME)
        run_demo_queries(model, index, sub_chunks_df)
        return
 
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
    # 1. Load chunks
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks_df = pd.read_parquet(CHUNKS_PATH)
    print(f"Loaded {len(chunks_df)} chunks.")
 
    # 2. Sub-chunk
    print("\nSub-chunking long sections...")
    sub_chunks_df = expand_chunks(chunks_df)
    print(f"Produced {len(sub_chunks_df)} sub-chunks.")
 
    # 3. Load embedding model
    print(f"\nLoading embedding model: {EMBED_MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(
        f"  Model loaded in {time.time() - t0:.1f}s. "
        f"Embedding dim: {model.get_sentence_embedding_dimension()}"
    )
 
    # 4. Embed
    print(f"\nEmbedding {len(sub_chunks_df)} sub-chunks "
          f"(batch_size={EMBED_BATCH_SIZE})...")
    t0 = time.time()
    embeddings = embed_texts(
        sub_chunks_df["chunk_text"].tolist(),
        model,
    )
    print(f"  Embedded in {time.time() - t0:.1f}s.  Shape: {embeddings.shape}")
 
    # 5. Build FAISS index
    print("\nBuilding FAISS IndexFlatIP...")
    index = build_faiss_index(embeddings)
    print(f"  Index size: {index.ntotal} vectors")
 
    # 6. Save
    print("\nSaving outputs...")
    sub_chunks_df.to_parquet(EMBEDDED_CHUNKS_PATH, index=False)
    np.save(EMBEDDINGS_PATH, embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"  - {EMBEDDED_CHUNKS_PATH}")
    print(f"  - {EMBEDDINGS_PATH}")
    print(f"  - {INDEX_PATH}")
 
    # 7. Sanity stats
    print_sanity_stats(chunks_df, sub_chunks_df, embeddings)
 
    # 8. Demo queries
    run_demo_queries(model, index, sub_chunks_df)
 
 
if __name__ == "__main__":
    main()