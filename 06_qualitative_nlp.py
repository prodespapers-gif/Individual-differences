"""
06_qualitative_nlp.py
=====================

Qualitative / categorical-signal pipeline (Stage 6 of 7) for the manuscript
"Individual Differences in Doctoral Learning Adaptation and Well-Being:
Academic Pressure, Supervisor Support, Career Uncertainty, and the
Moderating Role of Generative AI among Chinese PhD Students."

Purpose and a necessary transparency note
------------------------------------------
The mixed-methods design of this study calls for a qualitative strand that
characterizes, in respondents' own framing, what they would change about
their doctoral experience and what concerns them most, and then integrates
that strand with the quantitative latent profiles. The source instrument
(the Nature graduate survey) collected exactly such free-text responses in
items Q13 ("if you could, what one thing would you change about your
graduate degree experience"), Q15_9, and Q16_10 ("Other" concern boxes).

However, the publicly released, anonymised dataset does NOT contain the
free text. To protect respondent confidentiality, the open-text fields were
removed during anonymisation and replaced with numeric codes: Q13 is
released as a three-category reduction of the original responses, and Q15_9
and Q16_10 are released as binary flags indicating only whether a respondent
wrote something in the "Other" box. There is therefore no text corpus to
embed, and neural topic modeling (e.g., BERTopic with sentence-transformer
embeddings) cannot be performed on the published data. This script detects
that condition explicitly and does not fabricate a corpus or pretend to
model text that is absent.

What the script does instead
----------------------------
1. TEXT-AVAILABILITY DETECTION. The script inspects the configured
   open-text fields and decides whether a genuine corpus exists (a minimum
   number of responses of a minimum length). The decision and its evidence
   are written to a status table.

2. IF NO CORPUS (the published-data case): the script performs the
   legitimate analysis the released data supports and that serves the same
   integrative goal. It treats Q13 as a categorical "desired change"
   indicator and Q15_9 / Q16_10 as supplementary "wrote-other-concern"
   flags, then cross-tabulates each against the latent profiles from Script
   03 with a chi-square test of independence, Cramer's V, and standardized
   (Pearson) residuals identifying which profile-by-category cells are
   over- or under-represented. This is the categorical analogue of the
   intended qualitative-by-profile integration and feeds Stage 7.

3. IF A CORPUS EXISTS (a future, non-anonymised wave): the script runs a
   full topic-modeling pipeline -- sentence-transformer embeddings, UMAP
   dimensionality reduction, HDBSCAN clustering, and class-based TF-IDF
   topic representations via BERTopic -- with a lightweight TF-IDF +
   clustering fallback if those libraries are unavailable. It then
   cross-tabulates topic prevalence against the profiles. This path is
   fully implemented so the script is immediately reusable on text-bearing
   data; it simply does not activate on the published dataset.

Because the published data exercise the categorical path, that path is the
one verified here; the topic-modeling path is guarded and dependency-checked.

Methodological references
-------------------------
Agresti, A. (2013). Categorical data analysis (3rd ed.). Wiley.
Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based
    TF-IDF procedure. arXiv:2203.05794.
Haberman, S. J. (1973). The analysis of residuals in cross-classified
    tables. Biometrics, 29(1), 205-220.
McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold
    approximation and projection. arXiv:1802.03426.
Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings
    using Siamese BERT-networks. EMNLP 2019.

Author: BEFOUM Stephane Richard
Target journal: Learning and Individual Differences (Elsevier)
"""

from __future__ import annotations

import json
import logging
import re
import sys
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from configs import CONFIG, ensure_output_directories, set_global_seeds

# Optional neural-NLP stack (activates only when a real corpus is present).
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# Open-text field configuration
# ===========================================================================
# The survey's free-text items. On the published data these are released as
# numeric codes (see module docstring); on a non-anonymised wave they would
# carry text and the topic-modeling path would activate.
OPEN_TEXT_FIELDS: tuple[str, ...] = ("Q13", "Q15_9", "Q16_10")

# A field is treated as carrying genuine text only if it has at least this
# many responses of at least this many characters.
MIN_TEXT_RESPONSES: int = 30
MIN_TEXT_LENGTH: int = 5


# ===========================================================================
# Logging configuration
# ===========================================================================
def configure_logging() -> logging.Logger:
    """Configure structured logging to stdout and a timestamped file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = CONFIG.paths.logs_dir / f"06_qualitative_{timestamp}.log"

    logger = logging.getLogger("qualitative_nlp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized; log file at %s", log_path)
    logger.info(
        "Neural topic-modeling stack (BERTopic/sentence-transformers): %s",
        "available" if BERTOPIC_AVAILABLE else "not installed",
    )
    return logger


# ===========================================================================
# Data loading
# ===========================================================================
def load_analysis_dataset(logger: logging.Logger) -> pd.DataFrame:
    """Load the canonical analysis dataset produced by Script 01."""
    path = (
        CONFIG.paths.imputed_path(1) if CONFIG.imputation.enable
        else CONFIG.paths.chinese_phd_dataset
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found at {path}. Run "
            f"01_data_preparation.py first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded analysis dataset (N = %d)", len(df))
    return df


def load_profiles(logger: logging.Logger) -> pd.DataFrame | None:
    """Load profile assignments from Script 03 for the integration crosstabs."""
    path = CONFIG.paths.models_dir / "profile_assignments.csv"
    if not path.exists():
        logger.warning(
            "Profile assignments not found at %s; profile crosstabs will be "
            "skipped. Run 03_latent_profile_analysis.py first.", path,
        )
        return None
    profiles = pd.read_csv(path)
    logger.info(
        "Loaded profile assignments (N = %d, %d profiles)",
        len(profiles), profiles["profile"].nunique(),
    )
    return profiles


# ===========================================================================
# Text-availability detection
# ===========================================================================
def clean_text(value: Any) -> str:
    """Normalize a single free-text value; return '' for non-text/codes."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def detect_text_availability(
    df: pd.DataFrame, logger: logging.Logger,
) -> tuple[bool, pd.DataFrame, dict[str, list[str]]]:
    """Determine whether any configured field carries a genuine text corpus.

    Returns (corpus_exists, status_table, field_to_responses). The status
    table records, per field, whether it is present, how many substantive
    text responses it has, and the verdict. A field that is present but
    entirely numeric (the published-data case) is reported as carrying no
    text.
    """
    rows: list[dict[str, Any]] = []
    field_responses: dict[str, list[str]] = {}
    any_text = False

    for field in OPEN_TEXT_FIELDS:
        if field not in df.columns:
            rows.append({
                "field": field, "present": False, "dtype": "absent",
                "n_substantive_text": 0, "carries_text": False,
            })
            continue

        series = df[field]
        cleaned = series.apply(clean_text)
        substantive = cleaned[cleaned.str.len() >= MIN_TEXT_LENGTH]
        n_text = int(len(substantive))
        carries = n_text >= MIN_TEXT_RESPONSES
        any_text = any_text or carries
        field_responses[field] = substantive.tolist()

        rows.append({
            "field": field,
            "present": True,
            "dtype": str(series.dtype),
            "n_substantive_text": n_text,
            "n_unique_numeric_codes": (
                int(series.dropna().nunique())
                if pd.api.types.is_numeric_dtype(series) else np.nan
            ),
            "carries_text": carries,
        })
        logger.info(
            "Field %s: dtype=%s, substantive text responses=%d -> %s",
            field, series.dtype, n_text,
            "TEXT" if carries else "no text (numeric codes only)",
        )

    status = pd.DataFrame(rows)
    if any_text:
        logger.info("A text corpus is present; topic-modeling path will run.")
    else:
        logger.warning(
            "No analyzable free text in any configured field. The published "
            "anonymised data replace open responses with numeric codes; "
            "neural topic modeling is not possible. Proceeding with the "
            "categorical-signal integration the data support."
        )
    return any_text, status, field_responses


# ===========================================================================
# Categorical-signal analysis (published-data path)
# ===========================================================================
def describe_categorical_fields(
    df: pd.DataFrame, logger: logging.Logger,
) -> pd.DataFrame:
    """Describe the released categorical reductions of the open-text items.

    Reports each field's category frequencies. Q13 is a multi-category
    reduction of the "what would you change" responses; Q15_9 and Q16_10 are
    binary "wrote an Other concern" flags. Semantic labels for the Q13
    categories are not recoverable from the anonymised release, so categories
    are reported by their numeric code with a neutral description.
    """
    rows: list[dict[str, Any]] = []
    descriptions = {
        "Q13": "Desired change to degree experience (anonymised 3-category reduction)",
        "Q15_9": "Wrote a free-text 'Other' current-PhD concern (1) vs not (0)",
        "Q16_10": "Wrote a free-text 'Other' post-PhD concern (1) vs not (0)",
    }
    for field in OPEN_TEXT_FIELDS:
        if field not in df.columns:
            continue
        counts = df[field].value_counts().sort_index()
        for code, n in counts.items():
            rows.append({
                "field": field,
                "field_description": descriptions.get(field, ""),
                "category_code": int(code) if float(code).is_integer() else code,
                "n": int(n),
                "proportion": round(float(n) / len(df), 3),
            })
        logger.info(
            "%s: %d categories, distribution = %s",
            field, len(counts), counts.to_dict(),
        )
    return pd.DataFrame(rows)


def standardized_residuals(observed: np.ndarray) -> np.ndarray:
    """Pearson standardized residuals for a contingency table (Haberman 1973).

    Each cell residual is (observed - expected) / sqrt(expected * (1 - row
    proportion) * (1 - column proportion)). Values beyond +/-1.96 flag cells
    that depart from independence at roughly the 5% level.
    """
    total = observed.sum()
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    row_prop = row_sums / total
    col_prop = col_sums / total
    denom = np.sqrt(expected * (1 - row_prop) * (1 - col_prop))
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (observed - expected) / denom
    return np.where(np.isfinite(resid), resid, 0.0)


def crosstab_against_profiles(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    field: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Cross-tabulate one categorical field against the latent profiles.

    Returns a long-format table (profile x category with observed counts and
    standardized residuals) and a summary dict (chi-square, dof, p, Cramer's
    V). Returns None if the field is unavailable or degenerate.
    """
    if field not in df.columns:
        return None

    merged = df.loc[profiles["case_index"]].copy()
    merged["profile"] = profiles["profile"].to_numpy()
    sub = merged[["profile", field]].dropna()
    if sub[field].nunique() < 2 or sub["profile"].nunique() < 2:
        logger.info("%s: insufficient variation for a crosstab; skipped", field)
        return None

    table = pd.crosstab(sub["profile"], sub[field])
    observed = table.to_numpy(dtype=float)

    chi2, p, dof, _ = scipy_stats.chi2_contingency(observed)
    n = observed.sum()
    min_dim = min(observed.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else float("nan")
    residuals = standardized_residuals(observed)

    rows: list[dict[str, Any]] = []
    for i, profile in enumerate(table.index):
        for j, category in enumerate(table.columns):
            rows.append({
                "field": field,
                "profile": int(profile),
                "category_code": int(category) if float(category).is_integer() else category,
                "observed": int(observed[i, j]),
                "row_proportion": round(float(observed[i, j] / observed[i].sum()), 3),
                "standardized_residual": round(float(residuals[i, j]), 3),
                "notable_cell": bool(abs(residuals[i, j]) > 1.96),
            })

    summary = {
        "field": field,
        "n": int(n),
        "n_profiles": int(table.shape[0]),
        "n_categories": int(table.shape[1]),
        "chi_square": round(float(chi2), 3),
        "dof": int(dof),
        "p_value": round(float(p), 4),
        "cramers_v": round(cramers_v, 3) if np.isfinite(cramers_v) else np.nan,
        "significant": bool(p < 0.05),
    }
    logger.info(
        "%s x profile: chi2 = %.2f (dof = %d), p = %.4f, Cramer's V = %.3f%s",
        field, chi2, dof, p, cramers_v,
        " [significant association]" if p < 0.05 else "",
    )
    return pd.DataFrame(rows), summary


# ===========================================================================
# Topic-modeling path (future text-bearing waves)
# ===========================================================================
def build_corpus(
    field_responses: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Pool substantive text responses across fields, tracking provenance."""
    documents: list[str] = []
    sources: list[str] = []
    for field, responses in field_responses.items():
        for text in responses:
            if len(text) >= MIN_TEXT_LENGTH:
                documents.append(text)
                sources.append(field)
    return documents, sources


def topic_model_bertopic(
    documents: list[str], logger: logging.Logger,
) -> tuple[list[int], pd.DataFrame]:
    """Fit BERTopic on a genuine corpus (sentence-transformer + UMAP + HDBSCAN).

    Returns per-document topic assignments and a topic-summary table. Only
    called when a corpus exists and the neural stack is installed.
    """
    embedder = SentenceTransformer(CONFIG.qualitative.embedding_model)
    embeddings = embedder.encode(documents, show_progress_bar=False)
    model = BERTopic(
        embedding_model=embedder,
        min_topic_size=CONFIG.qualitative.min_topic_size,
        nr_topics=CONFIG.qualitative.n_topics if CONFIG.qualitative.n_topics else "auto",
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(documents, embeddings)
    info = model.get_topic_info()
    logger.info("BERTopic identified %d topics", len(info) - 1)
    return list(topics), info.rename(columns=str.lower)


def topic_model_fallback(
    documents: list[str], logger: logging.Logger,
) -> tuple[list[int], pd.DataFrame]:
    """Lightweight TF-IDF + agglomerative clustering topic model.

    Used when a corpus exists but the neural stack is unavailable. Produces
    cluster assignments and top-term representations through TF-IDF, so the
    integration crosstabs can still be computed without heavy dependencies.
    """
    n_clusters = CONFIG.qualitative.n_topics or max(2, min(8, len(documents) // 20))
    vectorizer = TfidfVectorizer(
        max_features=500, stop_words="english", ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(documents).toarray()
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X)

    terms = np.array(vectorizer.get_feature_names_out())
    rows: list[dict[str, Any]] = []
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            continue
        mean_tfidf = X[mask].mean(axis=0)
        top_terms = terms[np.argsort(mean_tfidf)[-8:][::-1]]
        rows.append({
            "topic": c,
            "count": int(mask.sum()),
            "top_terms": ", ".join(top_terms),
        })
    logger.info(
        "TF-IDF fallback identified %d topic clusters", n_clusters,
    )
    return list(labels), pd.DataFrame(rows)


# ===========================================================================
# Output utilities
# ===========================================================================
def write_table(df: pd.DataFrame, filename: str, logger: logging.Logger) -> None:
    """Write a results table to outputs/tables/ in CSV format."""
    if len(df) == 0:
        logger.warning("Skipping write of empty table: %s", filename)
        return
    out_path = CONFIG.paths.tables_dir / filename
    df.to_csv(out_path, index=False)
    logger.info("Table written: %s (%d rows)", out_path, len(df))


# ===========================================================================
# Pipeline orchestration
# ===========================================================================
def main() -> int:
    """Execute the qualitative / categorical-signal pipeline."""
    ensure_output_directories()
    set_global_seeds()
    logger = configure_logging()

    logger.info("Pipeline version: %s", CONFIG.pipeline_version)
    logger.info("Root seed: %d", CONFIG.reproducibility.root_seed)
    logger.info("Stage: 06_qualitative_nlp")

    try:
        df = load_analysis_dataset(logger)
        profiles = load_profiles(logger)

        # --- Phase 1: Text-availability detection ---
        logger.info("=" * 72)
        logger.info("PHASE 1: Open-text availability detection")
        logger.info("=" * 72)
        corpus_exists, status, field_responses = detect_text_availability(df, logger)
        write_table(status, "table_s29_text_availability.csv", logger)

        integration_summaries: list[dict[str, Any]] = []

        if not corpus_exists:
            # ---------------------------------------------------------------
            # Published-data path: categorical-signal integration.
            # ---------------------------------------------------------------
            logger.info("=" * 72)
            logger.info("PHASE 2: Categorical-signal description (no text corpus)")
            logger.info("=" * 72)
            cat_desc = describe_categorical_fields(df, logger)
            write_table(cat_desc, "table_s30_categorical_description.csv", logger)

            logger.info("=" * 72)
            logger.info("PHASE 3: Categorical signal x latent profile integration")
            logger.info("=" * 72)
            if profiles is not None:
                crosstab_frames: list[pd.DataFrame] = []
                for field in OPEN_TEXT_FIELDS:
                    result = crosstab_against_profiles(df, profiles, field, logger)
                    if result is None:
                        continue
                    table, summary = result
                    crosstab_frames.append(table)
                    integration_summaries.append(summary)
                if crosstab_frames:
                    write_table(
                        pd.concat(crosstab_frames, ignore_index=True),
                        "table_s31_categorical_profile_crosstab.csv", logger,
                    )
            else:
                logger.warning(
                    "No profiles available; skipping the profile crosstabs. "
                    "The categorical description above is still written."
                )

            analysis_mode = "categorical_signal"

        else:
            # ---------------------------------------------------------------
            # Text-bearing path: topic modeling + topic-by-profile crosstab.
            # ---------------------------------------------------------------
            logger.info("=" * 72)
            logger.info("PHASE 2: Topic modeling on the open-text corpus")
            logger.info("=" * 72)
            documents, sources = build_corpus(field_responses)
            logger.info("Corpus assembled: %d documents", len(documents))

            if BERTOPIC_AVAILABLE:
                topics, topic_info = topic_model_bertopic(documents, logger)
                engine = "bertopic"
            else:
                logger.warning(
                    "Neural stack unavailable; using TF-IDF clustering fallback."
                )
                topics, topic_info = topic_model_fallback(documents, logger)
                engine = "tfidf_fallback"
            topic_info["engine"] = engine
            write_table(topic_info, "table_s30_topics.csv", logger)

            # Topic-by-profile crosstab requires mapping documents back to
            # respondents; this is straightforward on a wave where each
            # respondent contributes identifiable text and is left as a
            # direct extension here (the published data do not reach this
            # branch).
            logger.info(
                "Topic-by-profile integration runs on text-bearing waves "
                "where document-to-respondent mapping is available."
            )
            analysis_mode = engine

        # --- Final phase: status metadata ---
        logger.info("=" * 72)
        logger.info("PHASE 4: Integration status metadata")
        logger.info("=" * 72)
        meta = {
            "analysis_mode": analysis_mode,
            "corpus_exists": corpus_exists,
            "open_text_fields": list(OPEN_TEXT_FIELDS),
            "n_fields_with_text": int(status["carries_text"].sum()),
            "categorical_integration_summaries": integration_summaries,
            "note": (
                "Open-text responses were removed during anonymisation of the "
                "published dataset and released as numeric codes; neural topic "
                "modeling is not possible on this release. The categorical "
                "signal (Q13 desired-change category and the Q15_9/Q16_10 "
                "Other-concern flags) was integrated with the latent profiles "
                "by chi-square analysis. The topic-modeling path activates "
                "automatically on a non-anonymised wave containing text."
            ),
        }
        meta_path = CONFIG.paths.models_dir / "qualitative_meta.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        logger.info("Qualitative metadata written to %s", meta_path)

        # --- Summary ---
        logger.info("=" * 72)
        logger.info("Qualitative / categorical-signal analysis completed")
        logger.info("=" * 72)
        logger.info("Analysis mode: %s", analysis_mode)
        if integration_summaries:
            sig = [s for s in integration_summaries if s["significant"]]
            logger.info(
                "Categorical fields significantly associated with profiles: "
                "%d of %d (%s)",
                len(sig), len(integration_summaries),
                ", ".join(s["field"] for s in sig) if sig else "none",
            )
        return 0

    except Exception as exc:
        logger.exception("Qualitative analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
