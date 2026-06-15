"""
01_data_preparation.py
======================

Data preparation pipeline (Stage 1 of 7) for the manuscript "Individual
Differences in Doctoral Learning Adaptation and Well-Being: Academic
Pressure, Supervisor Support, Career Uncertainty, and the Moderating Role
of Generative AI among Chinese PhD Students."

This script transforms the raw, anonymised Nature Careers Graduate Survey
2025 workbook into the single analysis-ready dataset that every downstream
script consumes. It produces:

  1. The canonical scored dataset (data/chinese_phd_dataset.csv).
  2. A harmonized JSON codebook documenting every derived variable, its
     scoring rule, and its verified value range (data/codebook.json).
  3. The Table 1 descriptive summary (outputs/tables/table_1_descriptives.csv).
  4. Optionally, M multiply-imputed datasets and an imputation-diagnostics
     file, but ONLY if CONFIG.imputation.enable is True. On the published
     data there is no item-level missingness, so imputation is off by
     default and this step is a deliberate, documented no-op.

Why this script departs from a generic survey pipeline
------------------------------------------------------
Three properties of the published dataset were verified directly against
the "Codes" worksheet and dictate the scoring:

  (a) Q15 (current PhD concerns) and Q16 (post-PhD concerns) are BINARY
      multiple-select checklists (0/1), not Likert ratings. Academic
      pressure and career uncertainty are therefore FORMATIVE indices,
      scored as the count of endorsed concerns within each block. They are
      not reverse-coded and not submitted to internal-consistency
      reliability or single-factor CFA (Bollen & Lennox, 1991).

  (b) The Likert blocks do not share a common maximum (Q12 is 1-5,
      Q20/Q33/Q34 are 1-6, Q11_New is 1-7, Q14a is 1-8). Reverse-coding
      is performed per item against that item's own maximum.

  (c) There is no item-level missingness on any analysis variable in the
      Chinese subset, so multiple imputation is disabled by default.

Reproducibility
---------------
The script is fully deterministic given configs.py. Running it twice on the
same raw input produces byte-identical outputs.

Methodological references
-------------------------
Bollen, K. A., & Lennox, R. (1991). Conventional wisdom on measurement.
    Psychological Bulletin, 110, 305-314.
Rubin, D. B. (1987). Multiple imputation for nonresponse in surveys. Wiley.
Schafer, J. L., & Graham, J. W. (2002). Missing data: Our view of the
    state of the art. Psychological Methods, 7(2), 147-177.
van Buuren, S. (2018). Flexible imputation of missing data (2nd ed.).
    Chapman and Hall/CRC.

Author: BEFOUM Stephane Richard
Target journal: Learning and Individual Differences (Elsevier)
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configs import (
    CONFIG,
    derive_imputation_seed,
    ensure_output_directories,
    set_global_seeds,
)

# Suppress non-actionable warnings from external libraries.
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning)


# ===========================================================================
# Logging configuration
# ===========================================================================
def configure_logging() -> logging.Logger:
    """Configure structured logging to both stdout and a timestamped file.

    The log file under outputs/logs/ provides the audit trail that
    reviewers can examine to verify every decision made during data
    preparation. The timestamp lets multiple runs coexist without
    overwriting one another.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = CONFIG.paths.logs_dir / f"01_data_preparation_{timestamp}.log"

    logger = logging.getLogger("data_preparation")
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
    return logger


# ===========================================================================
# Raw data loading and inclusion
# ===========================================================================
def load_nature_raw(logger: logging.Logger) -> pd.DataFrame:
    """Load the "Codes" worksheet of the raw Nature survey workbook.

    Only the numeric "Codes" worksheet is used. The "Labels" worksheet in
    the public release is offset by one column relative to "Codes" and is
    therefore unreliable for code-to-meaning alignment; the codebook built
    later in this script records item meaning from the verified "Codes"
    header positions instead.

    Raises
    ------
    FileNotFoundError
        If the raw workbook is not present at the configured location.
    """
    raw_path = CONFIG.paths.raw_nature_xlsx
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Nature raw data workbook not found at {raw_path}. "
            f"Download from Figshare (DOI: 10.6084/m9.figshare.30084739) "
            f"and place the .xlsx at this location before running."
        )

    logger.info("Loading Nature raw data from %s", raw_path)
    df = pd.read_excel(raw_path, sheet_name=CONFIG.paths.raw_nature_sheet)
    logger.info(
        "Loaded %d rows and %d columns from the '%s' worksheet",
        len(df), df.shape[1], CONFIG.paths.raw_nature_sheet,
    )

    expected_total_rows = 3785
    if len(df) != expected_total_rows:
        logger.warning(
            "Workbook row count (%d) differs from the expected %d. "
            "Verify the dataset version matches the Figshare release.",
            len(df), expected_total_rows,
        )
    return df


def apply_chinese_inclusion(
    df: pd.DataFrame, logger: logging.Logger,
) -> pd.DataFrame:
    """Extract the Chinese PhD subset using the dual inclusion criterion.

    Inclusion rule: Q7 == 36 (currently studying in China) OR Q6a == 36
    (Chinese origin, studying anywhere). On the published data this yields
    exactly N = 400. Three audit counts are logged and two audit columns
    are attached so that a reviewer can reconstruct either subpopulation.
    """
    code = CONFIG.study.china_country_code

    for required in ("Q7", "Q6a"):
        if required not in df.columns:
            raise KeyError(
                f"Country variable {required} is required for the inclusion "
                f"criterion but is missing from the raw worksheet."
            )

    in_china = df["Q7"] == code
    chinese_origin = df["Q6a"] == code

    n_in_china = int(in_china.sum())
    n_origin = int(chinese_origin.sum())
    n_origin_abroad = int((chinese_origin & ~in_china).sum())

    subset = df[in_china | chinese_origin].copy().reset_index(drop=True)
    n_combined = len(subset)

    # Audit columns preserved for downstream verification.
    subset["incl_studying_in_china"] = (
        in_china[in_china | chinese_origin].reset_index(drop=True).astype(int)
    )
    subset["incl_chinese_origin"] = (
        chinese_origin[in_china | chinese_origin]
        .reset_index(drop=True)
        .astype(int)
    )

    logger.info("Chinese inclusion criteria applied:")
    logger.info("  Studying in China (Q7 == 36):           n = %d", n_in_china)
    logger.info("  Chinese origin (Q6a == 36):             n = %d", n_origin)
    logger.info("  Chinese origin studying abroad:         n = %d", n_origin_abroad)
    logger.info("  Combined inclusion sample (union):      N = %d", n_combined)

    expected = CONFIG.study.expected_sample_size
    if n_combined != expected:
        logger.warning(
            "Combined sample N = %d differs from the expected N = %d. "
            "If the raw dataset has been updated, re-verify the inclusion "
            "logic against the new release.",
            n_combined, expected,
        )
    return subset


# ===========================================================================
# Reverse coding (reflective Likert items only, per-item scale maxima)
# ===========================================================================
def recode_reverse_items(
    df: pd.DataFrame, logger: logging.Logger,
) -> pd.DataFrame:
    """Reverse-code the reflective Likert items that require it.

    Reverse-coding maps x -> (scale_max + 1 - x) using each item's OWN
    maximum from CONFIG.study.likert_scale_max, so that for every reflective
    scale a higher score indicates more of the construct after recoding.
    The binary formative blocks (Q15, Q16) are never touched here; they are
    not in the reverse-coding registry by design.

    Missingness is preserved (NaN stays NaN through the arithmetic). Items
    not present in the dataset are logged and skipped rather than raising,
    so the registry may safely be a superset of any one dataset version.
    """
    df = df.copy()
    items_to_reverse = CONFIG.study.items_to_reverse
    scale_max = CONFIG.study.likert_scale_max

    logger.info(
        "Reverse-coding %d reflective Likert item(s) against per-item maxima",
        len(items_to_reverse),
    )

    n_recoded = 0
    for item in items_to_reverse:
        if item not in df.columns:
            logger.warning("Item %s not found; skipping reverse-code", item)
            continue
        if item not in scale_max:
            logger.warning(
                "Item %s has no registered scale maximum; skipping "
                "reverse-code to avoid an incorrect transformation",
                item,
            )
            continue
        max_value = scale_max[item]
        df[item] = max_value + 1 - df[item]
        n_recoded += 1
        logger.info("  Reversed %s against maximum %d", item, max_value)

    logger.info("Reverse coding complete: %d item(s) recoded", n_recoded)
    return df


# ===========================================================================
# Composite scoring
# ===========================================================================
def score_formative_index(
    df: pd.DataFrame,
    items: tuple[str, ...],
    name: str,
    logger: logging.Logger | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Score a formative binary index as a count of endorsed items.

    The Q15 and Q16 blocks are 0/1 multiple-select checklists. The index is
    the number of endorsed concerns (sum across the binary items), reported
    alongside the proportion endorsed. A respondent's index is NaN only if
    every constituent item is missing; otherwise missing entries within the
    block are treated as not-endorsed (0), which is the correct
    interpretation of an unchecked box in a select-all-that-apply item.

    Returns
    -------
    (count, proportion) : tuple of pandas.Series
        The endorsement count (0..len(items)) and the proportion (0..1).
    """
    available = [c for c in items if c in df.columns]
    missing = set(items) - set(available)
    if missing and logger is not None:
        logger.warning(
            "Formative index %s: items not found and excluded: %s",
            name, sorted(missing),
        )
    if not available:
        nan_series = pd.Series(np.nan, index=df.index, name=name)
        return nan_series, nan_series.rename(f"{name}_prop")

    block = df[available]
    all_missing = block.isna().all(axis=1)
    count = block.fillna(0).sum(axis=1)
    count = count.where(~all_missing, np.nan)
    proportion = count / len(available)
    count.name = name
    proportion.name = f"{name}_prop"

    if logger is not None:
        valid = int(count.notna().sum())
        logger.info(
            "Formative index %s: %d binary items, %d/%d valid, "
            "mean count = %.3f (SD = %.3f), mean proportion = %.3f",
            name, len(available), valid, len(df),
            float(count.mean()), float(count.std()), float(proportion.mean()),
        )
    return count, proportion


def score_reflective_scale(
    df: pd.DataFrame,
    items: tuple[str, ...],
    name: str,
    logger: logging.Logger | None = None,
    min_valid_items: int | None = None,
    standardize_items_first: bool = False,
) -> pd.Series:
    """Score a reflective Likert scale as a (optionally z-weighted) item mean.

    For scales whose items share a metric (Q20-only supervisor agreement,
    Q33 comfort, Q34 attitudes), a simple item mean is appropriate. For
    scales whose items live on DIFFERENT metrics (supervisor support mixes
    Q20 on 1-6 with Q14a on 1-8; well-being mixes Q11_New on 1-7 with Q12
    on 1-5), set ``standardize_items_first=True`` so each item is converted
    to a z-score before averaging, preventing the wider-range items from
    dominating the composite.

    A respondent's score is NaN if fewer than ``min_valid_items`` items are
    non-missing (Schafer & Graham, 2002). On the present data every item is
    complete, so this constraint is never binding here; it matters only on
    a future wave with missingness.
    """
    if min_valid_items is None:
        min_valid_items = max(1, len(items) // 2)

    available = [c for c in items if c in df.columns]
    missing = set(items) - set(available)
    if missing and logger is not None:
        logger.warning(
            "Reflective scale %s: items not found and excluded: %s",
            name, sorted(missing),
        )
    if not available:
        return pd.Series(np.nan, index=df.index, name=name)

    block = df[available].astype(float)
    if standardize_items_first:
        # Per-item z-scoring; guards against zero-variance items.
        means = block.mean(axis=0)
        sds = block.std(axis=0, ddof=0).replace(0.0, np.nan)
        block = (block - means) / sds

    n_valid = block.notna().sum(axis=1)
    composite = block.mean(axis=1, skipna=True)
    composite = composite.where(n_valid >= min_valid_items, np.nan)
    composite.name = name

    if logger is not None:
        valid = int(composite.notna().sum())
        logger.info(
            "Reflective scale %s: %d items%s, %d/%d valid, "
            "mean = %.3f (SD = %.3f)",
            name, len(available),
            " (item-standardized)" if standardize_items_first else "",
            valid, len(df), float(composite.mean()), float(composite.std()),
        )
    return composite


def zscore(series: pd.Series) -> pd.Series:
    """Return the z-score of a series; constant series map to all zeros."""
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index, name=series.name)
    return (series - series.mean()) / sd


def score_all_constructs(
    df: pd.DataFrame, logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Compute all construct scores and their standardized profile features.

    Produces, for the Chinese subset:
      - academic_pressure        formative count (0-6) + _prop
      - career_uncertainty       formative count (0-7) + _prop
      - supervisor_support       reflective, item-standardized mean (mixed metric)
      - wellbeing                reflective, item-standardized mean (mixed metric)
      - ai_comfort               reflective item mean (Q33, common 1-6 metric)
      - ai_concerns              reflective item mean (Q34, common 1-6 metric)
      - ai_use_ordinal           Q32 retained as ordinal AI-engagement (1-6)
      - ai_use                   binary AI-user flag (Q32 in positive codes)
      - <construct>_z            z-scored versions of the four focal constructs
                                 used as LPA profile features
    """
    s = CONFIG.study
    df = df.copy()

    # --- Formative binary indices ---
    df["academic_pressure"], df["academic_pressure_prop"] = score_formative_index(
        df, s.academic_pressure_items, "academic_pressure", logger,
    )
    df["career_uncertainty"], df["career_uncertainty_prop"] = score_formative_index(
        df, s.career_uncertainty_items, "career_uncertainty", logger,
    )

    # --- Reflective Likert scales ---
    # Supervisor support and well-being mix metrics, so item-standardize first.
    df["supervisor_support"] = score_reflective_scale(
        df, s.supervisor_support_items, "supervisor_support", logger,
        standardize_items_first=True,
    )
    df["wellbeing"] = score_reflective_scale(
        df, s.wellbeing_items, "wellbeing", logger,
        standardize_items_first=True,
    )
    # AI comfort and AI concerns each sit on a single common 1-6 metric.
    df["ai_comfort"] = score_reflective_scale(
        df, s.ai_comfort_items, "ai_comfort", logger,
        standardize_items_first=False,
    )
    df["ai_concerns"] = score_reflective_scale(
        df, s.ai_concern_items, "ai_concerns", logger,
        standardize_items_first=False,
    )

    # --- AI use: ordinal engagement plus a binary user flag ---
    if s.ai_use_item in df.columns:
        df["ai_use_ordinal"] = df[s.ai_use_item].astype("Float64")
        df["ai_use"] = (
            df[s.ai_use_item].isin(s.ai_use_positive_codes).astype("Int64")
        )
        if logger is not None:
            n_users = int((df["ai_use"] == 1).sum())
            logger.info(
                "AI use: %d/%d respondents flagged as AI users "
                "(Q32 in codes %s)",
                n_users, len(df), s.ai_use_positive_codes,
            )

    # --- Standardized profile features for LPA ---
    for construct in ("academic_pressure", "supervisor_support",
                      "career_uncertainty", "wellbeing"):
        df[f"{construct}_z"] = zscore(df[construct])

    return df


# ===========================================================================
# Optional multiple imputation (no-op by default on these data)
# ===========================================================================
@dataclass
class ImputationDiagnostic:
    """Per-variable diagnostic record for optional multiple imputation."""

    variable: str
    n_observed: int
    n_missing: int
    proportion_missing: float
    between_imputation_variance: float
    within_imputation_variance: float
    fraction_missing_information: float
    pooled_mean: float
    pooled_sd: float


def select_items_for_imputation(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify item-level numeric variables for the optional imputer.

    Returns the focal Likert items to impute and the demographic auxiliary
    predictors. The binary formative blocks are imputable too (they are
    numeric 0/1), but on the present data nothing is missing.
    """
    s = CONFIG.study
    items = list(
        s.academic_pressure_items
        + s.supervisor_support_items
        + s.career_uncertainty_items
        + s.wellbeing_items
        + s.ai_comfort_items
        + s.ai_concern_items
    )
    if s.ai_use_item in df.columns:
        items.append(s.ai_use_item)
    auxiliary = list(s.demographic_items)

    items = [c for c in items if c in df.columns]
    auxiliary = [c for c in auxiliary if c in df.columns]
    return items, auxiliary


def run_single_imputation(
    df: pd.DataFrame,
    items: list[str],
    auxiliary: list[str],
    imputation_index: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Generate one completed dataset using IterativeImputer (MICE).

    Imported lazily so the script has no hard dependency on the imputation
    stack when imputation is disabled. Each imputation uses a distinct,
    deterministic seed.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    cfg = CONFIG.imputation
    seed = derive_imputation_seed(imputation_index)

    df_imputed = df.copy()
    design_columns = items + auxiliary
    design_matrix = df_imputed[design_columns].to_numpy(dtype=float)

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=cfg.max_iter_per_imputation,
        random_state=seed,
        sample_posterior=cfg.sample_posterior,
        tol=cfg.tol,
        verbose=0,
    )
    completed = imputer.fit_transform(design_matrix)
    df_imputed[items] = completed[:, : len(items)]
    df_imputed["_imputation_index"] = imputation_index
    df_imputed["_imputation_seed"] = seed

    logger.info(
        "Imputation %d/%d complete (seed = %d)",
        imputation_index, cfg.n_imputations, seed,
    )
    return df_imputed


def compute_imputation_diagnostics(
    imputed_datasets: list[pd.DataFrame],
    observed_df: pd.DataFrame,
    items: list[str],
    logger: logging.Logger,
) -> list[ImputationDiagnostic]:
    """Compute Rubin's-rules diagnostics (between/within variance, FMI)."""
    M = len(imputed_datasets)
    if M < 2:
        logger.warning(
            "Only %d imputation(s); diagnostics require at least 2", M,
        )
        return []

    diagnostics: list[ImputationDiagnostic] = []
    for variable in items:
        if variable not in observed_df.columns:
            continue
        observed = observed_df[variable]
        n_observed = int(observed.notna().sum())
        n_missing = int(observed.isna().sum())
        prop_missing = n_missing / len(observed) if len(observed) else 0.0

        imp_means = np.array([float(d[variable].mean()) for d in imputed_datasets])
        imp_vars = np.array([float(d[variable].var(ddof=1)) for d in imputed_datasets])
        N = len(imputed_datasets[0])

        between = float(np.var(imp_means, ddof=1))
        within = float(np.mean(imp_vars) / N)
        total = within + (1 + 1 / M) * between
        fmi = ((1 + 1 / M) * between) / total if total > 0 else 0.0

        diagnostics.append(ImputationDiagnostic(
            variable=variable,
            n_observed=n_observed,
            n_missing=n_missing,
            proportion_missing=round(prop_missing, 4),
            between_imputation_variance=round(between, 6),
            within_imputation_variance=round(within, 6),
            fraction_missing_information=round(fmi, 4),
            pooled_mean=round(float(np.mean(imp_means)), 4),
            pooled_sd=round(float(np.sqrt(np.mean(imp_vars))), 4),
        ))

    logger.info("Computed imputation diagnostics for %d variables",
                len(diagnostics))
    return diagnostics


def run_optional_imputation(
    scored_df: pd.DataFrame, logger: logging.Logger,
) -> None:
    """Run multiple imputation only if enabled; otherwise document the no-op.

    On the published Chinese subset there is no item-level missingness, so
    CONFIG.imputation.enable is False and this function logs that imputation
    was intentionally skipped. When enabled (future wave / sensitivity
    analysis), it writes M completed datasets, re-scores constructs on each,
    and writes a diagnostics file.
    """
    items, auxiliary = select_items_for_imputation(scored_df)
    total_missing = int(scored_df[items].isna().sum().sum()) if items else 0

    if not CONFIG.imputation.enable:
        logger.info(
            "Multiple imputation DISABLED (CONFIG.imputation.enable = False). "
            "Item-level missing cells across focal items: %d. The analyses "
            "run on the single observed dataset.",
            total_missing,
        )
        if total_missing > 0:
            logger.warning(
                "Missing cells were detected (%d) but imputation is disabled. "
                "Set CONFIG.imputation.enable = True to handle missingness.",
                total_missing,
            )
        return

    logger.info(
        "Multiple imputation ENABLED: generating %d completed datasets",
        CONFIG.imputation.n_imputations,
    )
    imputed_datasets: list[pd.DataFrame] = []
    for m in range(1, CONFIG.imputation.n_imputations + 1):
        df_imp = run_single_imputation(scored_df, items, auxiliary, m, logger)
        df_imp = score_all_constructs(df_imp)  # re-score on completed data
        out_path = CONFIG.paths.imputed_path(m)
        df_imp.to_csv(out_path, index=False)
        imputed_datasets.append(df_imp)

    if CONFIG.imputation.compute_fmi:
        diagnostics = compute_imputation_diagnostics(
            imputed_datasets, scored_df, items, logger,
        )
        if diagnostics:
            rows = [d.__dict__ for d in diagnostics]
            pd.DataFrame(rows).to_csv(
                CONFIG.paths.imputation_diagnostics, index=False,
            )
            logger.info(
                "Imputation diagnostics written to %s",
                CONFIG.paths.imputation_diagnostics,
            )


# ===========================================================================
# Codebook generation
# ===========================================================================
def compute_value_ranges(
    df: pd.DataFrame, items: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Record the verified observed range and missingness for each item."""
    ranges: dict[str, dict[str, Any]] = {}
    for item in items:
        if item not in df.columns:
            continue
        series = df[item]
        non_null = series.dropna()
        ranges[item] = {
            "observed_min": (
                float(non_null.min()) if len(non_null) else None
            ),
            "observed_max": (
                float(non_null.max()) if len(non_null) else None
            ),
            "n_unique": int(non_null.nunique()),
            "n_missing": int(series.isna().sum()),
        }
    return ranges


def build_codebook(scored_df: pd.DataFrame) -> dict[str, Any]:
    """Construct the harmonized codebook documenting all variables.

    Item meaning is recorded from the verified "Codes" worksheet header
    positions (the "Labels" worksheet is column-shifted and not used). The
    codebook records, for every construct, whether it is formative or
    reflective, its scoring rule, its constituent items, and the verified
    value range of each item on the analysis sample.
    """
    s = CONFIG.study
    codebook: dict[str, Any] = {
        "manuscript_title": CONFIG.manuscript_title,
        "pipeline_version": CONFIG.pipeline_version,
        "generated_at": datetime.now().isoformat(),
        "root_seed": CONFIG.reproducibility.root_seed,
        "imputation_enabled": CONFIG.imputation.enable,
        "data_note": (
            "Item meaning recorded from the verified 'Codes' worksheet. "
            "The 'Labels' worksheet in the public release is offset by one "
            "column and was not used for code-to-meaning alignment. The "
            "Chinese subset has no item-level missingness on analysis "
            "variables."
        ),
        "sample": {
            "label": "Nature Careers Graduate Survey 2025 (Chinese subset)",
            "source": (
                "Springer Nature in partnership with Thinks Insight and "
                "Strategy"
            ),
            "doi": "10.6084/m9.figshare.30084739",
            "inclusion_rule": s.inclusion_rule,
            "expected_n": s.expected_sample_size,
            "actual_n": int(len(scored_df)),
            "data_collection_period": "15 May 2025 to 11 June 2025",
        },
        "constructs": {
            "academic_pressure": {
                "type": "formative_binary_index",
                "definition": (
                    "Load of current PhD concerns endorsed by the "
                    "respondent (funding, mentoring, finances, mental "
                    "health, publication pressure, imposter syndrome)."
                ),
                "items": list(s.academic_pressure_items),
                "scoring": (
                    "Count of endorsed binary (0/1) concerns; range 0-6. "
                    "Reported with proportion. Not reverse-coded; no "
                    "internal-consistency reliability or single-factor CFA "
                    "(formative indicators; Bollen & Lennox, 1991)."
                ),
                "item_value_ranges": compute_value_ranges(
                    scored_df, s.academic_pressure_items
                ),
            },
            "supervisor_support": {
                "type": "reflective_likert_scale",
                "definition": (
                    "Career-oriented and relational support from the "
                    "doctoral supervisor."
                ),
                "items": list(s.supervisor_support_items),
                "scoring": (
                    "Item-standardized mean (items span 1-6 and 1-8 "
                    "metrics; all items already keyed so higher = more "
                    "support in the anonymised release, so none reversed)."
                ),
                "reverse_coded_items": [
                    i for i in s.items_to_reverse
                    if i in s.supervisor_support_items
                ],
                "item_value_ranges": compute_value_ranges(
                    scored_df, s.supervisor_support_items
                ),
            },
            "career_uncertainty": {
                "type": "formative_binary_index",
                "definition": (
                    "Load of post-PhD concerns endorsed by the respondent "
                    "(qualification, guidance, job market, contract "
                    "precarity, finances, burnout, fulfillment)."
                ),
                "items": list(s.career_uncertainty_items),
                "scoring": (
                    "Count of endorsed binary (0/1) concerns; range 0-7. "
                    "Reported with proportion. Not reverse-coded; no "
                    "internal-consistency reliability or single-factor CFA."
                ),
                "item_value_ranges": compute_value_ranges(
                    scored_df, s.career_uncertainty_items
                ),
            },
            "wellbeing": {
                "type": "reflective_likert_scale",
                "definition": (
                    "Overall PhD satisfaction and experienced enjoyment, "
                    "fulfillment, and expectation alignment."
                ),
                "items": list(s.wellbeing_items),
                "scoring": (
                    "Item-standardized mean (Q11_New on 1-7 and Q12 items "
                    "on 1-5; all keyed so higher = better well-being, none "
                    "reversed)."
                ),
                "item_value_ranges": compute_value_ranges(
                    scored_df, s.wellbeing_items
                ),
            },
            "ai_moderators": {
                "type": "mixed",
                "definition": (
                    "Generative-AI use, comfort, and attitudes/concerns in "
                    "doctoral research practice."
                ),
                "items": {
                    "ai_use_item": s.ai_use_item,
                    "ai_use_positive_codes": list(s.ai_use_positive_codes),
                    "ai_comfort": list(s.ai_comfort_items),
                    "ai_concerns": list(s.ai_concern_items),
                },
                "scoring": (
                    "ai_use_ordinal: Q32 retained on its 1-6 metric. "
                    "ai_use: binary user flag (Q32 in positive codes). "
                    "ai_comfort and ai_concerns: item means on their common "
                    "1-6 metrics."
                ),
                "item_value_ranges": compute_value_ranges(
                    scored_df,
                    (s.ai_use_item,) + s.ai_comfort_items + s.ai_concern_items,
                ),
            },
        },
        "derived_profile_features": {
            "features": list(CONFIG.lpa.profile_features),
            "scoring": (
                "z-scores of academic_pressure, supervisor_support, "
                "career_uncertainty, and wellbeing on the analysis sample; "
                "used as the indicator vector for latent profile analysis."
            ),
        },
        "demographic_controls": {
            "items": list(s.demographic_items),
            "note": (
                "Q3a (PhD stage, 1-9) and Q3b (full-/part-time, 1-2) have "
                "unambiguous code semantics. Remaining demographics are "
                "used as categorical adjustment covariates under their code "
                "values without a substantive label claim, because the "
                "public 'Labels' worksheet is column-shifted."
            ),
            "item_value_ranges": compute_value_ranges(
                scored_df, s.demographic_items
            ),
        },
        "imputation": {
            "enabled": CONFIG.imputation.enable,
            "method": "MICE via sklearn IterativeImputer (when enabled)",
            "n_imputations": CONFIG.imputation.n_imputations,
            "rationale": (
                "Disabled by default: the published Chinese subset has no "
                "item-level missingness. Retained for future waves or "
                "sensitivity analyses."
            ),
        },
    }
    return codebook


def write_codebook(codebook: dict[str, Any], logger: logging.Logger) -> None:
    """Write the codebook to disk as UTF-8 JSON."""
    out_path = CONFIG.paths.codebook
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(codebook, fh, indent=2, ensure_ascii=False)
    logger.info("Codebook written to %s", out_path)


# ===========================================================================
# Descriptive summary (Table 1)
# ===========================================================================
def summarize_descriptives(
    df: pd.DataFrame, logger: logging.Logger,
) -> pd.DataFrame:
    """Produce the Table 1 descriptive summary.

    Covers the four focal constructs, the AI moderators, and the
    demographic controls. Continuous/derived variables are summarized by
    mean, SD, and range; categorical demographics by category frequencies.
    Computed on the analysis dataset (no imputation needed for description).
    """
    rows: list[dict[str, Any]] = []

    # Derived construct scores (continuous).
    continuous_vars = [
        "academic_pressure", "academic_pressure_prop",
        "supervisor_support",
        "career_uncertainty", "career_uncertainty_prop",
        "wellbeing",
        "ai_comfort", "ai_concerns", "ai_use_ordinal",
    ]
    for var in continuous_vars:
        if var not in df.columns:
            continue
        s = df[var].dropna().astype(float)
        if len(s) == 0:
            continue
        rows.append({
            "variable": var,
            "type": "continuous",
            "category": None,
            "n_valid": int(s.notna().sum()),
            "n": int(s.notna().sum()),
            "proportion": None,
            "mean": round(float(s.mean()), 4),
            "sd": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
        })

    # Binary AI-user flag.
    if "ai_use" in df.columns:
        s = df["ai_use"].dropna()
        for value, count in s.value_counts().sort_index().items():
            rows.append({
                "variable": "ai_use",
                "type": "binary",
                "category": str(int(value)),
                "n_valid": int(s.notna().sum()),
                "n": int(count),
                "proportion": round(float(count / len(df)), 4),
                "mean": None, "sd": None, "min": None, "max": None,
            })

    # Demographic controls (categorical, by code value).
    for item in CONFIG.study.demographic_items:
        if item not in df.columns:
            continue
        s = df[item].dropna()
        for value, count in s.value_counts().sort_index().items():
            rows.append({
                "variable": item,
                "type": "categorical",
                "category": str(value),
                "n_valid": int(s.notna().sum()),
                "n": int(count),
                "proportion": round(float(count / len(df)), 4),
                "mean": None, "sd": None, "min": None, "max": None,
            })

    summary = pd.DataFrame(rows)
    logger.info("Descriptive summary (Table 1): %d rows", len(summary))
    return summary


# ===========================================================================
# Pipeline orchestration
# ===========================================================================
def main() -> int:
    """Execute the full data-preparation pipeline."""
    ensure_output_directories()
    set_global_seeds()
    logger = configure_logging()

    logger.info("Pipeline version: %s", CONFIG.pipeline_version)
    logger.info("Root seed: %d", CONFIG.reproducibility.root_seed)
    logger.info("Target journal: %s", CONFIG.target_journal)
    logger.info("Stage: 01_data_preparation")

    try:
        # 1. Load and apply inclusion.
        logger.info("=" * 72)
        logger.info("Loading and filtering")
        logger.info("=" * 72)
        df_raw = load_nature_raw(logger)
        df_china = apply_chinese_inclusion(df_raw, logger)

        # 2. Reverse-code reflective items, then score all constructs.
        logger.info("=" * 72)
        logger.info("Scoring constructs")
        logger.info("=" * 72)
        df_recoded = recode_reverse_items(df_china, logger)
        df_scored = score_all_constructs(df_recoded, logger)

        # 3. Persist the canonical analysis dataset.
        out_path = CONFIG.paths.chinese_phd_dataset
        df_scored.to_csv(out_path, index=False)
        logger.info(
            "Canonical analysis dataset written to %s (N = %d, %d columns)",
            out_path, len(df_scored), df_scored.shape[1],
        )

        # 4. Optional multiple imputation (no-op on these data).
        logger.info("=" * 72)
        logger.info("Missing-data handling")
        logger.info("=" * 72)
        run_optional_imputation(df_scored, logger)

        # 5. Codebook.
        logger.info("=" * 72)
        logger.info("Codebook")
        logger.info("=" * 72)
        codebook = build_codebook(df_scored)
        write_codebook(codebook, logger)

        # 6. Table 1.
        logger.info("=" * 72)
        logger.info("Descriptive summary (Table 1)")
        logger.info("=" * 72)
        descriptives = summarize_descriptives(df_scored, logger)
        descriptives_path = CONFIG.paths.tables_dir / "table_1_descriptives.csv"
        descriptives.to_csv(descriptives_path, index=False)
        logger.info("Descriptive summary written to %s", descriptives_path)

        # Final summary.
        logger.info("=" * 72)
        logger.info("Data preparation completed successfully")
        logger.info("=" * 72)
        logger.info(
            "N = %d | academic_pressure mean = %.2f | "
            "career_uncertainty mean = %.2f | supervisor_support mean = %.2f | "
            "wellbeing mean = %.2f",
            len(df_scored),
            float(df_scored["academic_pressure"].mean()),
            float(df_scored["career_uncertainty"].mean()),
            float(df_scored["supervisor_support"].mean()),
            float(df_scored["wellbeing"].mean()),
        )
        return 0

    except Exception as exc:
        logger.exception("Data preparation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
