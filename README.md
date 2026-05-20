# NSCLC Clinical Trial RAG Assistant

A citation-grounded research assistant for **non-small cell lung cancer (NSCLC) immunotherapy and targeted therapy clinical trials**. Built to help oncology research assistants doing literature review answer questions like *"What biomarker testing is required for EGFR-targeted therapy enrollment?"* or *"Compare primary endpoints across pembrolizumab plus chemotherapy trials"* — with every claim cited back to either a ClinicalTrials.gov registry entry or a PubMed publication.

> **Disclaimer:** This is a research/portfolio project. It is **not** a medical device, does not provide medical advice, and refuses patient-specific enrollment recommendations.

---

## Headline results

Evaluated on a hand-labeled 20-question test set covering biomarker eligibility, endpoint comparison, cross-trial filtering, specific protocol lookup, and 5 adversarial questions (out-of-scope, patient-advice, speculative-future).

| Metric                                | Value     |
|---------------------------------------|-----------|
| **Lexical citation faithfulness**     | **0.994** |
| **Hit@5** (labeled retrieval questions) | **1.000** |
| **Recall@5**                          | **0.917** |
| **Recall@10**                         | **1.000** |
| **Mean Reciprocal Rank (MRR)**        | **1.000** |
| **Refusal correctness accuracy**      | **0.950** |
| Mean citations per answer             | 11.2      |
| False-positive refusals               | 0         |
| False-negative answers                | 1         |
| Hallucinated NCT identifiers          | 1 / ~168 cited (0.6%) |
| Hallucinated PMIDs                    | 0         |

Across ~168 total citations produced over 20 questions, exactly one referenced an NCT identifier not present in the retrieved chunks — surfaced and documented by the eval framework. The patient-advice pre-flight guard handled all three patient-specific questions deterministically in under 1 second each, with zero API spend.

---

## Architecture

```
ClinicalTrials.gov v2 API ──┐
                            ├─►  01_ingestData.py  ──►  data/processed/chunks.parquet
PubMed E-utilities API   ───┘                              │
                                                           ▼
                                              02_embedIndex.py
                                              (PubMedBERT + FAISS)
                                                           │
                                                           ▼
                                       data/processed/{embedded_chunks.parquet,
                                                       embeddings.npy,
                                                       faiss.index}
                                                           │
            ┌──────────────────────────────────────────────┤
            ▼                                              ▼
   05_streamlit_app.py                            03_generateRag.py
   (web UI, clickable                             (retrieval + generation
    citations)                                     + refusal gates)
                                                           │
                                                           ▼
                                                   04_evaluate.py
                                                   (eval harness)
```

**Pipeline summary:**

1. **Ingest** ~800 NSCLC immunotherapy/targeted-therapy trials from ClinicalTrials.gov v2 and their linked PubMed publications.
2. **Chunk** each record into semantic sections (eligibility inclusion/exclusion, primary outcomes, interventions, abstract sections, etc.) with section-aware metadata.
3. **Embed** every chunk with a biomedical-domain SentenceBERT model and index in FAISS.
4. **Retrieve** top-k chunks per query with metadata filtering, NCT auto-detection, optional source-type boosts, and similarity thresholding.
5. **Generate** an answer using Claude Sonnet under a strict system prompt requiring inline citations to structured source labels (`[S1 | NCT:... | PMID:... | section]`).
6. **Refuse** patient-specific advice via a deterministic regex pre-flight guard *before* any API call.

---

## Quickstart

### Requirements

- Python 3.10+
- ~1.5 GB free disk (model downloads + corpus)
- Internet access for first-time API pulls and model download
- An [Anthropic API key](https://console.anthropic.com) (~$5 of credit is more than enough)
- An [NCBI account](https://www.ncbi.nlm.nih.gov/account/) for the optional but recommended free API key

### Install

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
requests
pandas
pyarrow
biopython
sentence-transformers
faiss-cpu
anthropic
streamlit
python-dotenv
tqdm
```

### Configure `.env`

Create a `.env` file in the project root:

```
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=optional_but_bumps_you_from_3_to_10_req_per_second
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Run the full pipeline

```bash
# Phase 2: ingest ~800 trials and their linked PubMed publications
python 01_ingestData.py --max_trials 800

# Phase 3: embed and index (~5–15 min on CPU)
python 02_embedIndex.py

# Phase 4: ask one question via CLI
python 03_generateRag.py --question "What biomarker testing is required for EGFR-targeted therapy enrollment?" --show_context

# Phase 5: evaluate
python 04_evaluate.py --config_name baseline

# Phase 6: launch the web UI
streamlit run 05_streamlit_app.py
```

---

## Phase-by-phase details

### Phase 2 — Data ingestion (`01_ingestData.py`)

Pulls from **two APIs and joins on NCT IDs**:

- **ClinicalTrials.gov v2 REST API**: registry entries with structured fields (phase, status, eligibility, outcomes, interventions, sponsors). Filter scope: NSCLC condition + immunotherapy or targeted therapy intervention + start date ≥ 2018 + status ∉ {UNKNOWN, WITHDRAWN}.
- **PubMed via biopython's E-utilities**: publications linked to each NCT via two methods:
  - `NCT_ID[si]` Secondary Source ID search
  - PMID references already in CT.gov's `referencesModule` (typed as `RESULT`, `DERIVED`, or `BACKGROUND`)

**Provenance tracking.** Every publication chunk carries a `link_method` field indicating *how* it was joined to its trial (`pubmed_secondary_source_id`, `ctgov_result_or_derived_reference`, `ctgov_background_reference`). This lets downstream analysis ask "are papers found via CT.gov references more relevant than ones found via PubMed search?"

**Section-aware chunking.** Every trial becomes multiple chunks tagged by semantic section:
- `brief_summary`, `detailed_description`
- `eligibility_inclusion`, `eligibility_exclusion` (heuristic regex split; 97.5–98% match rate at scale)
- `primary_outcomes`, `secondary_outcomes`
- `interventions`, `arm_groups`
- `result_references`, `background_references`

Each publication adds chunks per structured abstract section (`abstract_background`, `abstract_methods`, `abstract_results`, `abstract_conclusion`, etc.) when labels are present in the PubMed XML.

**Resume cache.** PubMed lookups are cached per-NCT in `data/raw/pubmed/by_trial/<NCT>.json`. Re-runs skip already-fetched trials, so a crash at trial 600/800 doesn't reset the run.

**Snapshot timestamping.** Each record carries `pulled_at_utc` and `last_update_post_date`. CT.gov entries are mutable; the snapshot is reproducible.

**Corpus characteristics** (800 trials):
- 19% completed, 21% active not recruiting, 9% terminated → **49% likely results-bearing**
- 79% interventional, 21% observational (tagged via `evidence_type`)
- 18.4% have RESULT/DERIVED PMID references in their CT.gov record
- Median eligibility text length: 3,247 chars (P95: 11,125)

### Phase 3 — Embeddings and vector index (`02_embedIndex.py`)

**Embedding model: `pritamdeka/S-PubMedBert-MS-MARCO`.**
A BERT model pretrained on PubMed abstracts and fine-tuned on MS-MARCO for retrieval. Chosen over generic models (MiniLM, BGE) because biomedical jargon — *EGFR T790M, ECOG performance status, RECIST 1.1, PD-L1 TPS* — embeds meaningfully better with a domain-tuned model. 768-dim output, 512-token max input.

**Sub-chunking strategy.** Section-aware AND length-bounded:
- Long sections are split into overlapping windows of 1,000 chars with 200-char overlap.
- Sentence-boundary search looks back up to 150 chars from the tentative window end and prefers cutting at `.`, `!`, or `?` followed by whitespace.
- All section metadata (section name, NCT, PMID, source type, MeSH terms) is preserved on every sub-chunk.

**Vector store: FAISS `IndexFlatIP` with L2-normalized vectors.**
Inner product on unit vectors equals cosine similarity. Exact (not approximate) search, because the corpus is small enough (~thousands of chunks) that ANN structures like HNSW or IVF would be premature optimization. Above ~100k vectors, swap in HNSW.

**Persistence.** Three artifacts: `embedded_chunks.parquet` (sub-chunks + metadata), `embeddings.npy` (raw float32 vectors), and `faiss.index`. The FAISS index can always be rebuilt from `embeddings.npy` without re-running the embedding model, which is the slow step (~15 min on CPU for ~5,000 sub-chunks).

### Phase 4 — Retrieval and generation (`03_generateRag.py`)

**Retrieval pipeline:**

1. **NCT auto-detection.** If the query contains `NCT\d{8}`, the retriever bypasses FAISS entirely and returns all chunks for that NCT. Pure-semantic retrieval is bad at exact-identifier lookups; this special-case routing fixes the failure cleanly.
2. **Over-fetch + filter + light reranking.** FAISS returns top-50, then filters by `source_type` and `nct_id`, applies optional additive boosts to publication chunks or registry chunks, drops anything below `min_score = 0.75`, and returns top-k (default 8).
3. **Structured source labels.** Every retrieved chunk gets a label like `[S1 | NCT:NCT04105153 | PMID:22285168 | pubmed_publication | abstract_interpretation]`. These are formatted into the prompt and the model is required to use them verbatim for citations.

**Generation:**

- **Model:** Claude Sonnet at temperature 0.1.
- **System prompt** enforces (a) no outside knowledge, (b) cite every factual claim, (c) distinguish registry-protocol claims from published-results claims, (d) refuse patient-specific advice, (e) refuse when retrieved context is out-of-scope, (f) prefer precision over confidence.
- **Required answer format:** Direct answer → Evidence (bulleted, all cited) → Registry vs publication distinction → Limitations / uncertainty.

**Three layers of refusal:**

1. **Pre-flight patient-advice guard** — deterministic regex match against patterns like *"should I/my X enroll"*, *"my mother/father/spouse/patient"*, *"recommend a trial for me"*. Fires before any API call, returns a fixed refusal string in <100 ms.
2. **Retrieval gate** — if fewer than 2 chunks pass the similarity threshold, refuse with no API call.
3. **In-prompt scope check** — system prompt rule directs the model to refuse when the question's topic is clearly outside the retrieved context's domain (e.g., user asks about breast cancer but retrieval returns NSCLC trials).

### Phase 5 — Evaluation (`04_evaluate.py`)

Custom eval harness over a hand-labeled JSONL question set. Custom metrics rather than RAGAS to avoid the LangChain dependency and to make every metric directly inspectable.

**Metrics computed per question:**
- **Retrieval:** Hit@5, Hit@10, Recall@5, Recall@10, MRR (against hand-labeled expected NCT IDs).
- **Citation faithfulness:** lexical check — for every `[Sn | NCT:... | PMID:...]` label parsed out of the generated answer, verify the NCT and PMID actually appear in the retrieved chunks. Anything else is a hallucinated identifier.
- **Refusal correctness:** 4-cell confusion matrix (TP correct refusal / TN correct answer / FP incorrect refusal / FN incorrect answer). FN is the worst category for medical RAG (system answered when it should have refused).

**A/B-ready.** Every run takes a `--config_name` flag and writes `results_<config>_<ts>.jsonl` + `summary_<config>_<ts>.json`. Configuration changes (`--prefer_publications`, `--prefer_registry`, `--min_score 0.65`, etc.) can be compared head-to-head over the same question set.

**Eval question categories (n=20):**
biomarker eligibility (2), common eligibility (3), endpoint comparison (3), cross-trial filter (2), specific protocol (1), specific biomarker (1), endpoint results (1), trial design (1), safety (1), out-of-scope (1), out-of-scope-future (1), patient advice (3).

### Phase 6 — Streamlit UI (`05_streamlit_app.py`)

**Standout feature: clickable citations.** Every `[S1 | NCT:... | PMID:... | ...]` in the generated answer is linkified into an actual hyperlink that opens the source on PubMed or ClinicalTrials.gov. Every claim is one click from primary evidence.

Sidebar exposes all retrieval and Claude knobs (top-k, fetch-k, min similarity, source filter, boost toggles, NCT ID filter, temperature, max tokens). Source cards render each retrieved chunk in an expander with full metadata, similarity score, and chunk text. Raw answer text and raw retrieved DataFrame are available in collapsibles for debugging.

---

## Design decisions worth defending

| Decision | Why |
|---|---|
| Biomedical embedding model over generic | A/B-able interview moment: biomedical jargon embeds better with PubMedBERT than with MiniLM/BGE. |
| Section-aware + length-bounded sub-chunking | Pure fixed-size chunking loses semantic structure; pure section chunking blows past the 512-token embedding ceiling. |
| FAISS exact (Flat) search | Corpus size doesn't justify ANN approximation. Exact retrieval is faster *and* more accurate at this scale. |
| Multi-method PubMed linking with provenance | Roughly doubles publication yield vs `[si]` search alone; `link_method` enables downstream A/B analysis. |
| Structured citation labels in prompt and output | Lets the eval harness regex-parse citations and verify every identifier against retrieval — that's how 0.994 faithfulness is measurable, not just claimed. |
| Deterministic pre-flight patient-advice guard | System prompt rules can be overridden when retrieval is strong. Regex match is non-bypassable and saves API spend. |
| NCT exact-match retrieval bypass | Semantic embeddings retrieve by meaning, not by string. NCT IDs carry no semantic content. Bypassing FAISS when an NCT is in the query is the cleanest fix. |
| Custom eval metrics over RAGAS | Transparency: every metric is ~20 lines of inspectable Python rather than a black-box library call. No LangChain dependency. |
| UNKNOWN/WITHDRAWN status filter | These trials never enrolled or haven't been updated in years. Filtering them at ingest improved the results-bearing fraction of the corpus from 31% to 49%. |

---

## Known limitations

1. **Speculative future-event questions.** Questions like *"What is the next FDA-approved drug for NSCLC?"* aren't well-handled by the current scope check — the topic is in-scope (NSCLC) but the temporal framing requires prediction. This is the one remaining FN in the eval and a documented v2 priority.
2. **Lexical faithfulness only.** The eval verifies cited NCT/PMID identifiers exist in retrieval. It does NOT verify the cited claim is actually supported by the cited chunk's text. A semantic-faithfulness check using an LLM judge would catch this and is on the roadmap.
3. **Retrieval label coverage.** Only 3 of 20 eval questions have hand-labeled `expected_nct_ids`. Hit@5 / Recall@5 / MRR are statistically meaningful but noisy. Labeling the remaining 17 questions is straightforward but time-consuming.
4. **No hybrid search.** Pure-semantic retrieval. Adding BM25 sparse retrieval with reciprocal rank fusion would likely improve niche-vocabulary queries (gene names, specific protocol identifiers) — measured improvement is a v2 experiment.
5. **No response streaming in UI.** Claude's 15-second generation latency is hidden behind a spinner. Streaming would dramatically improve perceived speed.
6. **Snapshot, not live.** Trial data is pulled at ingestion time. The system does not refresh as CT.gov entries are updated. Each record carries `last_update_post_date` for transparency.

---

## Roadmap (v2 priorities)

- Hybrid sparse + dense retrieval (BM25 + FAISS via reciprocal rank fusion).
- Cross-encoder reranking of the top-50 (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) before passing to Claude.
- LLM-as-judge semantic faithfulness check (verify each (claim, cited chunk) pair).
- Streaming generation in the Streamlit UI.
- Anthropic prompt caching on the system prompt to cut per-call costs ~50%.
- Expand the eval set to 50+ questions with full retrieval labels across all categories.
- Speculative/future-event question detector to close the remaining FN.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data ingestion | `requests`, `biopython.Entrez`, ClinicalTrials.gov v2 API, PubMed E-utilities, PMC OA |
| Processing | `pandas`, `pyarrow` (Parquet) |
| Embeddings | `sentence-transformers`, `pritamdeka/S-PubMedBert-MS-MARCO` |
| Vector search | `faiss-cpu` (IndexFlatIP) |
| Generation | `anthropic` SDK, Claude Sonnet |
| UI | `streamlit` |
| Configuration | `python-dotenv`, `argparse` |
| Progress / observability | `tqdm`, custom sanity stats |

---

## Project layout

```
.
├── 01_ingestData.py            # CT.gov + PubMed ingestion -> chunks.parquet
├── 02_embedIndex.py            # Sub-chunk + embed + FAISS index
├── 03_generateRag.py           # Retrieve + Claude generation + refusal guards
├── 04_evaluate.py              # Eval harness over hand-labeled JSONL
├── 05_streamlit_app.py         # Web UI
├── requirements.txt
├── .env                        # NCBI + Anthropic keys (gitignored)
└── data/
    ├── raw/
    │   ├── clinicaltrials/
    │   └── pubmed/
    │       └── by_trial/       # per-NCT resume cache
    ├── processed/
    │   ├── chunks.parquet
    │   ├── embedded_chunks.parquet
    │   ├── embeddings.npy
    │   └── faiss.index
    └── eval/
        ├── eval_set.jsonl
        ├── results_<config>_<ts>.jsonl
        └── summary_<config>_<ts>.json
```

---

## Acknowledgments

- ClinicalTrials.gov for the open v2 API and JATS-format registry data.
- NCBI E-utilities and PubMed Central for free programmatic access to biomedical literature.
- Prithivida Deka et al. for the `S-PubMedBert-MS-MARCO` model on HuggingFace.
- Anthropic for Claude Sonnet and the API access that made the generation layer practical.

---

## License

MIT.

## Final Result

<img width="1919" height="934" alt="image" src="https://github.com/user-attachments/assets/b0de0774-2c1c-4754-a48c-492ccc5208f3" />
