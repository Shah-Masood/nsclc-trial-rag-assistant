from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

DATA_DIR = Path("data")
RAW_CT_DIR = DATA_DIR / "raw" / "clinicaltrials"
RAW_PUBMED_DIR = DATA_DIR / "raw" / "pubmed"
RAW_PUBMED_BY_TRIAL_DIR = RAW_PUBMED_DIR / "by_trial"
PROCESSED_DIR = DATA_DIR / "processed"

for path in [RAW_CT_DIR, RAW_PUBMED_DIR, RAW_PUBMED_BY_TRIAL_DIR, PROCESSED_DIR]:
    path.mkdir(parents=True, exist_ok=True)


IMMUNOTHERAPY_TERMS = [
    "pembrolizumab",
    "nivolumab",
    "atezolizumab",
    "durvalumab",
    "cemiplimab",
    "ipilimumab",
    "tremelimumab",
    "PD-1",
    "PD-L1",
    "CTLA-4",
    "immune checkpoint",
    "immunotherapy",
]

TARGETED_THERAPY_TERMS = [
    "osimertinib",
    "erlotinib",
    "gefitinib",
    "afatinib",
    "dacomitinib",
    "amivantamab",
    "lazertinib",
    "alectinib",
    "brigatinib",
    "lorlatinib",
    "crizotinib",
    "selpercatinib",
    "pralsetinib",
    "sotorasib",
    "adagrasib",
    "trastuzumab deruxtecan",
    "EGFR",
    "ALK",
    "ROS1",
    "BRAF",
    "MET",
    "RET",
    "NTRK",
    "KRAS",
    "HER2",
    "targeted therapy",
]


# -----------------------------
# General helpers
# -----------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_get(obj: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur = obj

    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]

    return cur


def flatten_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return " ".join(flatten_text(v) for v in value if v is not None).strip()

    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values()).strip()

    return str(value).strip()


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def contains_any(text: str, terms: List[str]) -> bool:
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)


def normalize_phase_value(value: str) -> str:
    return str(value or "").lower().replace(" ", "").replace("-", "").replace("/", "_")


def classify_evidence_type(study_type: Optional[str], phases: List[str]) -> str:
    study_type_text = str(study_type or "").upper()

    if "OBSERVATIONAL" in study_type_text:
        return "observational_or_real_world"

    if phases:
        return "interventional_trial"

    return "unclear_or_unphased"


def list_to_clean_string(values: Any) -> str:
    if not values:
        return ""

    if isinstance(values, list):
        return ", ".join(str(v) for v in values if v)

    return str(values)


# -----------------------------
# PubMed cache helpers
# -----------------------------

def pubmed_trial_cache_path(nct_id: str) -> Path:
    return RAW_PUBMED_BY_TRIAL_DIR / f"{nct_id}.json"


def load_cached_pubmed_for_trial(nct_id: str) -> Optional[List[Dict[str, Any]]]:
    path = pubmed_trial_cache_path(nct_id)

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cached_pubmed_for_trial(nct_id: str, records: List[Dict[str, Any]]) -> None:
    path = pubmed_trial_cache_path(nct_id)

    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)


# -----------------------------
# ClinicalTrials.gov ingestion
# -----------------------------

def fetch_ctgov_studies(
    max_trials: int = 100,
    page_size: int = 100,
    sleep_seconds: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Pull NSCLC trials from ClinicalTrials.gov v2.

    Broad API query, then local filtering for:
    - 2018+ start date
    - NSCLC relevance
    - immunotherapy or targeted therapy terms
    - no UNKNOWN / WITHDRAWN status
    """

    studies: List[Dict[str, Any]] = []
    next_page_token: Optional[str] = None
    fetched_count = 0

    params: Dict[str, Any] = {
        "format": "json",
        "pageSize": page_size,
        "query.cond": "Non-Small Cell Lung Cancer",
        "query.term": (
            "AREA[StartDate]RANGE[2018-01-01,MAX] "
            "AND (immunotherapy OR pembrolizumab OR nivolumab OR atezolizumab "
            "OR durvalumab OR osimertinib OR EGFR OR ALK OR targeted therapy)"
        ),
    }

    while len(studies) < max_trials:
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(CTGOV_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        batch = payload.get("studies", [])

        if not batch:
            break

        fetched_count += len(batch)

        for study in batch:
            if is_relevant_trial(study):
                studies.append(study)

                if len(studies) >= max_trials:
                    break

        next_page_token = payload.get("nextPageToken")

        if not next_page_token:
            break

        time.sleep(sleep_seconds)

    print(f"CT.gov records fetched before local filtering: {fetched_count}")
    print(
        f"CT.gov local filter pass rate: {len(studies)}/{fetched_count}"
        if fetched_count
        else "No records fetched."
    )

    return studies


def is_relevant_trial(study: Dict[str, Any]) -> bool:
    protocol = study.get("protocolSection", {})

    identification = protocol.get("identificationModule", {})
    design = protocol.get("designModule", {})
    status = protocol.get("statusModule", {})
    arms = protocol.get("armsInterventionsModule", {})
    conditions = protocol.get("conditionsModule", {})
    description = protocol.get("descriptionModule", {})

    nct_id = identification.get("nctId", "")
    phases = design.get("phases", []) or []
    start_date = safe_get(status, ["startDateStruct", "date"], "")
    overall_status = status.get("overallStatus")

    if not nct_id:
        return False

    # Remove dead-weight records before scaling.
    if overall_status in {"UNKNOWN", "WITHDRAWN"}:
        return False

    all_text = flatten_text({
        "conditions": conditions,
        "arms": arms,
        "briefTitle": identification.get("briefTitle"),
        "officialTitle": identification.get("officialTitle"),
        "briefSummary": description.get("briefSummary"),
        "detailedDescription": description.get("detailedDescription"),
    })

    phase_text = " ".join(normalize_phase_value(p) for p in phases)
    phase_ok = any(
        p in phase_text
        for p in ["phase2", "phase3", "phase1_phase2", "phase2_phase3"]
    )

    # Keep NO_PHASE records so observational/RWE studies can be tagged.
    # Drop Phase 1-only and other irrelevant phase records.
    if phases and not phase_ok:
        return False

    if start_date and start_date[:4].isdigit():
        if int(start_date[:4]) < 2018:
            return False

    therapy_ok = contains_any(all_text, IMMUNOTHERAPY_TERMS + TARGETED_THERAPY_TERMS)
    condition_ok = contains_any(all_text, ["non-small cell lung", "nsclc", "lung cancer"])

    return therapy_ok and condition_ok


def parse_ctgov_references(protocol: Dict[str, Any]) -> List[Dict[str, Any]]:
    references_module = protocol.get("referencesModule", {})
    references = []

    for ref in references_module.get("references", []) or []:
        references.append({
            "pmid": ref.get("pmid"),
            "type": ref.get("type"),
            "citation": ref.get("citation"),
        })

    return references


def parse_ctgov_study(study: Dict[str, Any], pulled_at: str) -> Dict[str, Any]:
    protocol = study.get("protocolSection", {})

    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    description = protocol.get("descriptionModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    conditions = protocol.get("conditionsModule", {})
    arms = protocol.get("armsInterventionsModule", {})
    outcomes = protocol.get("outcomesModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})

    nct_id = identification.get("nctId")
    phases = design.get("phases", []) or []
    study_type = design.get("studyType")
    references = parse_ctgov_references(protocol)

    interventions = []
    for intervention in arms.get("interventions", []) or []:
        interventions.append({
            "name": intervention.get("name"),
            "type": intervention.get("type"),
            "description": intervention.get("description"),
            "arm_group_labels": intervention.get("armGroupLabels", []),
        })

    arm_groups = []
    for arm in arms.get("armGroups", []) or []:
        arm_groups.append({
            "label": arm.get("label"),
            "type": arm.get("type"),
            "description": arm.get("description"),
            "intervention_names": arm.get("interventionNames", []),
        })

    primary_outcomes = []
    for outcome in outcomes.get("primaryOutcomes", []) or []:
        primary_outcomes.append({
            "measure": outcome.get("measure"),
            "description": outcome.get("description"),
            "time_frame": outcome.get("timeFrame"),
        })

    secondary_outcomes = []
    for outcome in outcomes.get("secondaryOutcomes", []) or []:
        secondary_outcomes.append({
            "measure": outcome.get("measure"),
            "description": outcome.get("description"),
            "time_frame": outcome.get("timeFrame"),
        })

    result_reference_pmids = [
        ref.get("pmid")
        for ref in references
        if ref.get("pmid") and str(ref.get("type", "")).upper() in {"RESULT", "DERIVED"}
    ]

    background_reference_pmids = [
        ref.get("pmid")
        for ref in references
        if ref.get("pmid") and str(ref.get("type", "")).upper() == "BACKGROUND"
    ]

    return {
        "nct_id": nct_id,
        "source_type": "clinicaltrials_registry",
        "pulled_at_utc": pulled_at,
        "last_update_post_date": safe_get(status, ["lastUpdatePostDateStruct", "date"]),
        "brief_title": identification.get("briefTitle"),
        "official_title": identification.get("officialTitle"),
        "brief_summary": description.get("briefSummary"),
        "detailed_description": description.get("detailedDescription"),
        "overall_status": status.get("overallStatus"),
        "start_date": safe_get(status, ["startDateStruct", "date"]),
        "completion_date": safe_get(status, ["completionDateStruct", "date"]),
        "phases": phases,
        "study_type": study_type,
        "evidence_type": classify_evidence_type(study_type, phases),
        "enrollment_count": safe_get(design, ["enrollmentInfo", "count"]),
        "conditions": conditions.get("conditions", []),
        "eligibility_criteria": eligibility.get("eligibilityCriteria"),
        "sex": eligibility.get("sex"),
        "minimum_age": eligibility.get("minimumAge"),
        "maximum_age": eligibility.get("maximumAge"),
        "healthy_volunteers": eligibility.get("healthyVolunteers"),
        "interventions": interventions,
        "arm_groups": arm_groups,
        "primary_outcomes": primary_outcomes,
        "secondary_outcomes": secondary_outcomes,
        "lead_sponsor": safe_get(sponsor, ["leadSponsor", "name"]),
        "collaborators": sponsor.get("collaborators", []),
        "references": references,
        "result_reference_pmids": result_reference_pmids,
        "background_reference_pmids": background_reference_pmids,
    }


# -----------------------------
# PubMed ingestion
# -----------------------------

def setup_entrez() -> None:
    load_dotenv()

    email = os.getenv("NCBI_EMAIL")
    api_key = os.getenv("NCBI_API_KEY")

    if not email:
        raise ValueError(
            "Missing NCBI_EMAIL. Add it to your .env file. "
            "Example .env: NCBI_EMAIL=your_email@example.com"
        )

    Entrez.email = email
    Entrez.tool = "nsclc_trial_ingestion_phase2"

    if api_key:
        Entrez.api_key = api_key


def entrez_sleep_seconds() -> float:
    return 0.15 if os.getenv("NCBI_API_KEY") else 0.40


def pubmed_search_by_nct(nct_id: str, max_pmids: int = 20) -> List[str]:
    term = f"{nct_id}[si]"

    with Entrez.esearch(
        db="pubmed",
        term=term,
        retmax=max_pmids,
        sort="pub date",
    ) as handle:
        record = Entrez.read(handle)

    return list(record.get("IdList", []))


def fetch_pubmed_records(pmids: List[str]) -> List[Dict[str, Any]]:
    if not pmids:
        return []

    unique_pmids = sorted(set(str(pmid) for pmid in pmids if pmid))

    if not unique_pmids:
        return []

    with Entrez.efetch(
        db="pubmed",
        id=",".join(unique_pmids),
        retmode="xml",
    ) as handle:
        records = Entrez.read(handle, validate=False)

    return records.get("PubmedArticle", [])


def extract_article_ids(pubmed_article: Dict[str, Any]) -> Dict[str, Optional[str]]:
    ids = {
        "pmid": None,
        "pmcid": None,
        "doi": None,
    }

    citation = pubmed_article.get("MedlineCitation", {})
    article_ids = pubmed_article.get("PubmedData", {}).get("ArticleIdList", [])

    pmid_obj = citation.get("PMID")

    if pmid_obj:
        ids["pmid"] = str(pmid_obj)

    for aid in article_ids:
        id_type = aid.attributes.get("IdType") if hasattr(aid, "attributes") else None
        value = str(aid)

        if id_type == "pmc":
            ids["pmcid"] = value
        elif id_type == "doi":
            ids["doi"] = value
        elif id_type == "pubmed":
            ids["pmid"] = value

    return ids


def parse_pubmed_article(
    pubmed_article: Dict[str, Any],
    nct_id: str,
    pulled_at: str,
    link_method: str,
) -> Dict[str, Any]:
    citation = pubmed_article.get("MedlineCitation", {})
    article = citation.get("Article", {})
    journal = article.get("Journal", {})

    article_ids = extract_article_ids(pubmed_article)
    title = flatten_text(article.get("ArticleTitle"))

    abstract_sections = []
    abstract_obj = article.get("Abstract", {})
    abstract_texts = abstract_obj.get("AbstractText", []) if abstract_obj else []

    for part in abstract_texts:
        label = None

        if hasattr(part, "attributes"):
            label = part.attributes.get("Label") or part.attributes.get("NlmCategory")

        abstract_sections.append({
            "section": label or "UNLABELED_ABSTRACT",
            "text": str(part),
        })

    abstract_full = "\n".join(
        f"{s['section']}: {s['text']}" if s["section"] else s["text"]
        for s in abstract_sections
    ).strip()

    pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
    year = pub_date.get("Year")
    medline_date = pub_date.get("MedlineDate")

    mesh_terms = []
    for heading in citation.get("MeshHeadingList", []) or []:
        descriptor = heading.get("DescriptorName")

        if descriptor:
            mesh_terms.append(str(descriptor))

    databank_nct_ids = []
    databank_list = article.get("DataBankList", [])

    for databank in databank_list or []:
        accession_numbers = databank.get("AccessionNumberList", [])

        for accession in accession_numbers:
            accession_str = str(accession)

            if accession_str.startswith("NCT"):
                databank_nct_ids.append(accession_str)

    return {
        "nct_id": nct_id,
        "source_type": "pubmed_publication",
        "link_method": link_method,
        "pulled_at_utc": pulled_at,
        "pmid": article_ids["pmid"],
        "pmcid": article_ids["pmcid"],
        "doi": article_ids["doi"],
        "title": title,
        "journal": flatten_text(journal.get("Title")),
        "publication_year": year or medline_date,
        "abstract": abstract_full,
        "abstract_sections": abstract_sections,
        "mesh_terms": mesh_terms,
        "databank_nct_ids": databank_nct_ids,
    }


# -----------------------------
# RAG chunk creation
# -----------------------------

ELIGIBILITY_SPLIT_STATS = Counter()


def split_eligibility(criteria: Optional[str]) -> Tuple[str, str]:
    if not criteria:
        ELIGIBILITY_SPLIT_STATS["missing"] += 1
        return "", ""

    text = criteria.strip()

    inclusion_match = re.search(
        r"(inclusion criteria\s*:?.*?)(exclusion criteria\s*:?.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if inclusion_match:
        ELIGIBILITY_SPLIT_STATS["matched"] += 1
        inclusion = inclusion_match.group(1).strip()
        exclusion = inclusion_match.group(2).strip()
        return inclusion, exclusion

    ELIGIBILITY_SPLIT_STATS["fallback_no_exclusion_marker"] += 1
    return text, ""


def outcomes_to_text(outcomes: List[Dict[str, Any]], label: str) -> str:
    lines = []

    for i, outcome in enumerate(outcomes or [], start=1):
        measure = outcome.get("measure") or "Not specified"
        time_frame = outcome.get("time_frame") or "Not specified"
        description = outcome.get("description") or ""

        lines.append(
            f"{label} {i}: {measure}. "
            f"Time frame: {time_frame}. "
            f"Description: {description}"
        )

    return "\n".join(lines)


def interventions_to_text(interventions: List[Dict[str, Any]]) -> str:
    lines = []

    for i, intervention in enumerate(interventions or [], start=1):
        name = intervention.get("name") or "Not specified"
        intervention_type = intervention.get("type") or "Not specified"
        description = intervention.get("description") or ""
        arms = list_to_clean_string(intervention.get("arm_group_labels") or [])

        lines.append(
            f"Intervention {i}: {name}. "
            f"Type: {intervention_type}. "
            f"Arm groups: {arms}. "
            f"Description: {description}"
        )

    return "\n".join(lines)


def arm_groups_to_text(arm_groups: List[Dict[str, Any]]) -> str:
    lines = []

    for i, arm in enumerate(arm_groups or [], start=1):
        label = arm.get("label") or "Not specified"
        arm_type = arm.get("type") or "Not specified"
        description = arm.get("description") or ""
        interventions = list_to_clean_string(arm.get("intervention_names") or [])

        lines.append(
            f"Arm group {i}: {label}. "
            f"Type: {arm_type}. "
            f"Interventions: {interventions}. "
            f"Description: {description}"
        )

    return "\n".join(lines)


def references_to_text(
    references: List[Dict[str, Any]],
    wanted_types: Optional[set[str]] = None,
) -> str:
    lines = []

    for i, ref in enumerate(references or [], start=1):
        ref_type = str(ref.get("type") or "UNKNOWN").upper()

        if wanted_types and ref_type not in wanted_types:
            continue

        pmid = ref.get("pmid") or "No PMID"
        citation = ref.get("citation") or ""

        lines.append(
            f"Reference {i}: Type: {ref_type}. PMID: {pmid}. Citation: {citation}"
        )

    return "\n".join(lines)


def make_chunks_from_trial(trial: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []

    base_meta = {
        "nct_id": trial["nct_id"],
        "pmid": None,
        "pmcid": None,
        "source_type": "clinicaltrials_registry",
        "evidence_type": trial.get("evidence_type"),
        "study_type": trial.get("study_type"),
        "overall_status": trial.get("overall_status"),
        "phases": trial.get("phases", []),
        "pulled_at_utc": trial["pulled_at_utc"],
        "last_update_post_date": trial["last_update_post_date"],
        "title": trial["brief_title"] or trial["official_title"],
        "conditions": trial.get("conditions", []),
        "mesh_terms": [],
        "link_method": None,
    }

    inclusion, exclusion = split_eligibility(trial.get("eligibility_criteria"))

    registry_sections = [
        ("brief_summary", trial.get("brief_summary")),
        ("detailed_description", trial.get("detailed_description")),
        ("eligibility_inclusion", inclusion),
        ("eligibility_exclusion", exclusion),
        ("primary_outcomes", outcomes_to_text(trial.get("primary_outcomes", []), "Primary outcome")),
        ("secondary_outcomes", outcomes_to_text(trial.get("secondary_outcomes", []), "Secondary outcome")),
        ("interventions", interventions_to_text(trial.get("interventions", []))),
        ("arm_groups", arm_groups_to_text(trial.get("arm_groups", []))),
        ("result_references", references_to_text(trial.get("references", []), {"RESULT", "DERIVED"})),
        ("background_references", references_to_text(trial.get("references", []), {"BACKGROUND"})),
    ]

    for section_name, text in registry_sections:
        text = flatten_text(text)

        if text:
            chunks.append({
                **base_meta,
                "section_name": section_name,
                "chunk_text": text,
                "chunk_char_length": len(text),
            })

    return chunks


def make_chunks_from_publication(pub: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []

    base_meta = {
        "nct_id": pub["nct_id"],
        "pmid": pub["pmid"],
        "pmcid": pub["pmcid"],
        "source_type": "pubmed_publication",
        "evidence_type": "published_literature",
        "study_type": None,
        "overall_status": None,
        "phases": [],
        "pulled_at_utc": pub["pulled_at_utc"],
        "last_update_post_date": None,
        "title": pub["title"],
        "conditions": [],
        "mesh_terms": pub.get("mesh_terms", []),
        "link_method": pub.get("link_method"),
    }

    sections = pub.get("abstract_sections", [])

    if sections:
        for section in sections:
            text = flatten_text(section.get("text"))

            if text:
                section_label = str(section.get("section") or "unlabeled").lower()
                section_label = re.sub(r"[^a-z0-9_]+", "_", section_label)

                chunks.append({
                    **base_meta,
                    "section_name": f"abstract_{section_label}",
                    "chunk_text": text,
                    "chunk_char_length": len(text),
                })

    elif pub.get("abstract"):
        text = flatten_text(pub["abstract"])

        chunks.append({
            **base_meta,
            "section_name": "abstract",
            "chunk_text": text,
            "chunk_char_length": len(text),
        })

    return chunks


# -----------------------------
# Sanity stats
# -----------------------------

def print_sanity_stats(
    trials_df: pd.DataFrame,
    publications_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> None:
    print("\nQuick sanity stats:")
    print(f"- Trials pulled: {len(trials_df)}")
    print(f"- PubMed publications pulled: {len(publications_df)}")
    print(f"- RAG chunks created: {len(chunks_df)}")

    if not trials_df.empty:
        print("\nStatus split:")
        print(trials_df["overall_status"].fillna("MISSING").value_counts().to_string())

        print("\nEvidence type split:")
        print(trials_df["evidence_type"].fillna("MISSING").value_counts().to_string())

        print("\nStudy type split:")
        print(trials_df["study_type"].fillna("MISSING").value_counts().to_string())

        print("\nPhase split:")
        phase_counter = Counter()

        for phases in trials_df["phases"]:
            if isinstance(phases, list) and phases:
                phase_counter.update(phases)
            else:
                phase_counter.update(["NO_PHASE"])

        print(pd.Series(dict(phase_counter)).sort_values(ascending=False).to_string())

        trials_with_ctgov_result_refs = trials_df[
            trials_df["result_reference_pmids"].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            )
        ]["nct_id"].nunique()

        print(
            f"\nTrials with CT.gov RESULT/DERIVED reference PMIDs: "
            f"{trials_with_ctgov_result_refs}"
        )

    if not publications_df.empty:
        linked_trial_count = publications_df["nct_id"].nunique()
        print(f"Trials with linked PubMed publications: {linked_trial_count}")

        print("\nPubMed link method split:")
        print(publications_df["link_method"].fillna("MISSING").value_counts().to_string())

    else:
        print("Trials with linked PubMed publications: 0")

    print("\nEligibility splitter stats:")
    for key, value in ELIGIBILITY_SPLIT_STATS.items():
        print(f"- {key}: {value}")

    if not chunks_df.empty:
        print("\nChunk length stats, characters:")
        print(chunks_df["chunk_char_length"].describe().to_string())

        longest = chunks_df.sort_values("chunk_char_length", ascending=False).head(3)
        print("\nLongest chunks:")

        for _, row in longest.iterrows():
            print(
                f"- {row.get('nct_id')} | {row.get('source_type')} | "
                f"{row.get('section_name')} | chars={row.get('chunk_char_length')}"
            )


# -----------------------------
# PubMed collection
# -----------------------------

def collect_pmids_for_trial(
    trial: Dict[str, Any],
    max_pmids_per_trial: int,
) -> Dict[str, List[str]]:
    nct_id = trial["nct_id"]

    pmids_by_method = {
        "pubmed_secondary_source_id": [],
        "ctgov_result_or_derived_reference": [],
        "ctgov_background_reference": [],
    }

    if nct_id:
        pmids_by_method["pubmed_secondary_source_id"] = pubmed_search_by_nct(
            nct_id,
            max_pmids=max_pmids_per_trial,
        )

    pmids_by_method["ctgov_result_or_derived_reference"] = [
        str(pmid)
        for pmid in trial.get("result_reference_pmids", [])
        if pmid
    ]

    pmids_by_method["ctgov_background_reference"] = [
        str(pmid)
        for pmid in trial.get("background_reference_pmids", [])
        if pmid
    ]

    return pmids_by_method


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(max_trials: int, page_size: int, max_pmids_per_trial: int) -> None:
    pulled_at = now_utc_iso()
    filename_ts = utc_timestamp_for_filename()

    print(f"Pull timestamp UTC: {pulled_at}")
    print(f"Fetching up to {max_trials} ClinicalTrials.gov records...")

    raw_studies = fetch_ctgov_studies(max_trials=max_trials, page_size=page_size)

    raw_ct_path = RAW_CT_DIR / f"ctgov_nsclc_raw_{filename_ts}.jsonl"
    write_jsonl(raw_ct_path, raw_studies)

    print(f"Saved raw CT.gov records: {raw_ct_path}")
    print(f"Relevant CT.gov studies found: {len(raw_studies)}")

    trials = [parse_ctgov_study(study, pulled_at) for study in raw_studies]
    trials_df = pd.DataFrame(trials)

    setup_entrez()

    all_pubmed_articles_raw = []
    publications = []
    seen_pub_records = set()

    print("Searching PubMed by NCT ID and CT.gov reference PMIDs...")

    for trial in tqdm(trials, desc="PubMed lookup"):
        nct_id = trial["nct_id"]

        if not nct_id:
            continue

        cached_publications = load_cached_pubmed_for_trial(nct_id)

        if cached_publications is not None:
            for pub in cached_publications:
                pub_key = (pub.get("nct_id"), pub.get("pmid"))

                if pub_key not in seen_pub_records:
                    seen_pub_records.add(pub_key)
                    publications.append(pub)

            continue

        trial_publications = []

        try:
            pmids_by_method = collect_pmids_for_trial(
                trial,
                max_pmids_per_trial=max_pmids_per_trial,
            )

            for link_method, pmids in pmids_by_method.items():
                unique_pmids = sorted(set(str(pmid) for pmid in pmids if pmid))

                if not unique_pmids:
                    continue

                raw_articles = fetch_pubmed_records(unique_pmids)

                for raw_article in raw_articles:
                    parsed = parse_pubmed_article(
                        raw_article,
                        nct_id=nct_id,
                        pulled_at=pulled_at,
                        link_method=link_method,
                    )

                    pub_key = (parsed.get("nct_id"), parsed.get("pmid"))

                    if pub_key in seen_pub_records:
                        continue

                    seen_pub_records.add(pub_key)

                    all_pubmed_articles_raw.append({
                        "nct_id": nct_id,
                        "link_method": link_method,
                        "raw_pubmed_article": raw_article,
                    })

                    publications.append(parsed)
                    trial_publications.append(parsed)

                time.sleep(entrez_sleep_seconds())

            save_cached_pubmed_for_trial(nct_id, trial_publications)

        except Exception as e:
            print(f"Warning: PubMed failed for {nct_id}: {e}")

        time.sleep(entrez_sleep_seconds())

    raw_pubmed_path = RAW_PUBMED_DIR / f"pubmed_nsclc_raw_{filename_ts}.jsonl"

    raw_pubmed_serializable = [
        {
            "nct_id": rec["nct_id"],
            "link_method": rec["link_method"],
            "raw_pubmed_article": json.loads(
                json.dumps(rec["raw_pubmed_article"], default=str)
            ),
        }
        for rec in all_pubmed_articles_raw
    ]

    write_jsonl(raw_pubmed_path, raw_pubmed_serializable)

    publications_df = pd.DataFrame(publications)

    chunks = []

    for trial in trials:
        chunks.extend(make_chunks_from_trial(trial))

    for pub in publications:
        chunks.extend(make_chunks_from_publication(pub))

    chunks_df = pd.DataFrame(chunks)

    trials_path = PROCESSED_DIR / "trials.parquet"
    publications_path = PROCESSED_DIR / "publications.parquet"
    chunks_path = PROCESSED_DIR / "chunks.parquet"

    trials_df.to_parquet(trials_path, index=False)
    publications_df.to_parquet(publications_path, index=False)
    chunks_df.to_parquet(chunks_path, index=False)

    print("\nDone.")
    print(f"Trials saved:        {trials_path}        rows={len(trials_df)}")
    print(f"Publications saved:  {publications_path}  rows={len(publications_df)}")
    print(f"Chunks saved:        {chunks_path}        rows={len(chunks_df)}")
    print(f"Raw CT.gov saved:    {raw_ct_path}")
    print(f"Raw PubMed saved:    {raw_pubmed_path}")

    print_sanity_stats(trials_df, publications_df, chunks_df)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_trials", type=int, default=100)
    parser.add_argument("--page_size", type=int, default=100)
    parser.add_argument("--max_pmids_per_trial", type=int, default=20)

    args = parser.parse_args()

    run_pipeline(
        max_trials=args.max_trials,
        page_size=args.page_size,
        max_pmids_per_trial=args.max_pmids_per_trial,
    )


if __name__ == "__main__":
    main()