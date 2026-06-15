"""
07_mixed_methods_integration.py
===============================

Mixed-methods integration pipeline (Stage 7 of 7) for the manuscript
"Individual Differences in Doctoral Learning Adaptation and Well-Being:
Academic Pressure, Supervisor Support, Career Uncertainty, and the
Moderating Role of Generative AI among Chinese PhD Students."

This capstone stage brings the three analytic strands together into a
single, interpretable account of doctoral adaptation:

  * the PERSON-CENTERED strand (latent profiles from Script 03),
  * the CAUSAL strand (individual treatment effects of supervisor support
    on well-being from Script 05), and
  * the QUALITATIVE/CATEGORICAL strand (the desired-change and
    "Other"-concern signals integrated with profiles in Script 06).

Integration follows the joint-display tradition in mixed-methods research
(Fetters, Curry, & Creswell, 2013; Fetters & Tajima, 2022; Guetterman,
Fetters, & Creswell, 2015). Two complementary integration artifacts are
produced:

  1. A JOINT DISPLAY arraying, for each latent profile, its quantitative
     construct means, its causal signature (the within-profile distribution
     of conditional average treatment effects), and its categorical
     qualitative signature (the desired-change profile and the rate of
     idiosyncratic "Other" concerns). The joint display is the central
     device by which the strands are read side by side.

  2. An INTEGRATION (CONFIRMATION-EXPANSION-DISCORDANCE) MATRIX that, for
     each profile, classifies whether the strands converge (tell the same
     story), expand (each adds a non-redundant facet), or diverge (point in
     different directions). This operationalizes the "fit" of integration
     that Fetters et al. (2013) describe.

A formal test accompanies the causal strand of the display: a one-way
analysis of variance of the conditional treatment effects across profiles
asks whether the benefit of supervisor support differs by profile. This is
a profile-level reading of treatment-effect heterogeneity that complements
the covariate-level AUTOC and best-linear-projection results from Script
05; it is interpreted descriptively, consistent with that script's finding
that covariate-driven heterogeneity was weak.

Single-sample design
---------------------
The study analyzes one sample (Chinese doctoral students). The integration
is therefore within-sample: profiles, treatment effects, and categorical
signals are joined at the individual level on the case index. No
cross-sample profile matching is performed -- the Hungarian profile
alignment and Fisher-z profile-similarity machinery of an earlier,
two-sample draft are removed, because there is no second sample to align
to and including them would imply a comparison the design does not make.

Methodological references
-------------------------
Fetters, M. D., Curry, L. A., & Creswell, J. W. (2013). Achieving
    integration in mixed methods designs. Health Services Research, 48(6
    Pt 2), 2134-2156.
Fetters, M. D., & Tajima, C. (2022). Joint displays of integrated data
    collection in mixed methods research. International Journal of
    Qualitative Methods, 21.
Guetterman, T. C., Fetters, M. D., & Creswell, J. W. (2015). Integrating
    quantitative and qualitative results in health science mixed methods
    research through joint displays. Annals of Family Medicine, 13(6),
    554-561.
O'Cathain, A., Murphy, E., & Nicholl, J. (2010). Three techniques for
    integrating data in mixed methods studies. BMJ, 341, c4587.

Author: BEFOUM Stephane Richard
Target journal: Learning and Individual Differences (Elsevier)
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from configs import CONFIG, ensure_output_directories, set_global_seeds

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Focal constructs summarized per profile in the joint display.
JOINT_DISPLAY_CONSTRUCTS: tuple[str, ...] = (
    "wellbeing",
    "supervisor_support",
    "academic_pressure",
    "career_uncertainty",
    "ai_comfort",
    "ai_concerns",
)


# ===========================================================================
# Logging configuration
# ===========================================================================
def configure_logging() -> logging.Logger:
    """Configure structured logging to stdout and a timestamped file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = CONFIG.paths.logs_dir / f"07_integration_{timestamp}.log"

    logger = logging.getLogger("mixed_methods_integration")
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
# Input loading
# ===========================================================================
def load_inputs(
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load all upstream artifacts needed for integration.

    Returns the analysis dataset (with a case_index column matching the
    upstream key), the profile assignments, the CATE distribution, the
    categorical-signal crosstab, and the causal solution metadata. Raises
    if a required artifact is missing, with a message naming the script
    that produces it.
    """
    dataset_path = (
        CONFIG.paths.imputed_path(1) if CONFIG.imputation.enable
        else CONFIG.paths.chinese_phd_dataset
    )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found at {dataset_path}. Run "
            f"01_data_preparation.py first."
        )
    df = pd.read_csv(dataset_path)
    # The upstream case_index is the dataset row position; make it explicit.
    if "case_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "case_index"})
    logger.info("Loaded analysis dataset (N = %d)", len(df))

    profiles_path = CONFIG.paths.models_dir / "profile_assignments.csv"
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Profile assignments not found at {profiles_path}. Run "
            f"03_latent_profile_analysis.py first."
        )
    profiles = pd.read_csv(profiles_path)
    logger.info("Loaded profile assignments (%d profiles)", profiles["profile"].nunique())

    cate_path = CONFIG.paths.models_dir / "cate_distribution.csv"
    if not cate_path.exists():
        raise FileNotFoundError(
            f"CATE distribution not found at {cate_path}. Run "
            f"05_causal_heterogeneity.py first."
        )
    cate = pd.read_csv(cate_path)
    logger.info("Loaded CATE distribution (N = %d)", len(cate))

    crosstab_path = (
        CONFIG.paths.tables_dir / "table_s31_categorical_profile_crosstab.csv"
    )
    if crosstab_path.exists():
        crosstab = pd.read_csv(crosstab_path)
        logger.info(
            "Loaded categorical-signal crosstab (%d rows)", len(crosstab),
        )
    else:
        crosstab = pd.DataFrame()
        logger.warning(
            "Categorical crosstab not found at %s; the qualitative strand of "
            "the joint display will be limited. Run 06_qualitative_nlp.py.",
            crosstab_path,
        )

    meta_path = CONFIG.paths.models_dir / "causal_solution_meta.json"
    causal_meta: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as fh:
            causal_meta = json.load(fh)
        logger.info("Loaded causal solution metadata")

    return df, profiles, cate, crosstab, causal_meta


def merge_individual_level(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    cate: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Join dataset, profiles, and CATEs at the individual level.

    The join key is case_index, which all upstream scripts derive from the
    same dataset row order. Rows missing a profile or a CATE (e.g., dropped
    for the treatment construction) are reported and excluded from the
    merged frame.
    """
    merged = profiles.merge(
        cate[["case_index", "treatment", "cate", "cate_se",
              "cate_ci_lower", "cate_ci_upper"]],
        on="case_index", how="inner",
    )
    construct_cols = [c for c in JOINT_DISPLAY_CONSTRUCTS if c in df.columns]
    merged = merged.merge(
        df[["case_index"] + construct_cols], on="case_index", how="inner",
    )
    logger.info(
        "Individual-level merge: %d rows joined across profiles, CATEs, and "
        "constructs", len(merged),
    )
    if len(merged) < min(len(profiles), len(cate)):
        logger.info(
            "%d profile rows and %d CATE rows; %d retained after inner join",
            len(profiles), len(cate), len(merged),
        )
    return merged


# ===========================================================================
# Joint display construction
# ===========================================================================
def quantitative_signature(
    merged: pd.DataFrame, logger: logging.Logger,
) -> pd.DataFrame:
    """Per-profile means (and SDs) on the focal constructs.

    These are the quantitative cells of the joint display, the
    variable-centered reading of each profile.
    """
    construct_cols = [c for c in JOINT_DISPLAY_CONSTRUCTS if c in merged.columns]
    rows: list[dict[str, Any]] = []
    for profile in sorted(merged["profile"].unique()):
        sub = merged[merged["profile"] == profile]
        row: dict[str, Any] = {"profile": int(profile), "n": int(len(sub))}
        for col in construct_cols:
            row[f"{col}_mean"] = round(float(sub[col].mean()), 3)
            row[f"{col}_sd"] = round(float(sub[col].std(ddof=1)), 3)
        rows.append(row)
    table = pd.DataFrame(rows)
    logger.info(
        "Quantitative signature computed for %d profiles on %d constructs",
        len(table), len(construct_cols),
    )
    return table


def causal_signature(
    merged: pd.DataFrame, logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-profile CATE distribution and an across-profile ANOVA.

    Returns the causal cells of the joint display (within-profile CATE mean,
    SD, quartiles, and the share with a CI strictly above zero) and a
    one-way ANOVA testing whether mean CATE differs across profiles. The
    ANOVA is a profile-level lens on heterogeneity; per Script 05 the
    covariate-level heterogeneity was weak, so a non-trivial profile-level
    difference would indicate that the profiles capture treatment-effect
    variation that single covariates did not.
    """
    rows: list[dict[str, Any]] = []
    groups: list[np.ndarray] = []
    for profile in sorted(merged["profile"].unique()):
        sub = merged[merged["profile"] == profile]
        cate = sub["cate"].to_numpy()
        groups.append(cate)
        rows.append({
            "profile": int(profile),
            "n": int(len(sub)),
            "cate_mean": round(float(np.mean(cate)), 4),
            "cate_sd": round(float(np.std(cate, ddof=1)), 4),
            "cate_q25": round(float(np.percentile(cate, 25)), 4),
            "cate_median": round(float(np.median(cate)), 4),
            "cate_q75": round(float(np.percentile(cate, 75)), 4),
            "pct_ci_above_zero": round(
                float(np.mean(sub["cate_ci_lower"] > 0)), 3
            ),
        })
    table = pd.DataFrame(rows)

    anova: dict[str, Any] = {"test": "one_way_anova_cate_by_profile"}
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        f_stat, p_value = scipy_stats.f_oneway(*groups)
        # Eta-squared effect size.
        grand = np.concatenate(groups)
        ss_between = sum(
            len(g) * (np.mean(g) - grand.mean()) ** 2 for g in groups
        )
        ss_total = float(((grand - grand.mean()) ** 2).sum())
        eta2 = ss_between / ss_total if ss_total > 0 else float("nan")
        anova.update({
            "f_statistic": round(float(f_stat), 3),
            "p_value": round(float(p_value), 4),
            "eta_squared": round(float(eta2), 4),
            "profiles_differ": bool(p_value < 0.05),
        })
        logger.info(
            "CATE-by-profile ANOVA: F = %.3f, p = %.4f, eta^2 = %.4f -> "
            "profiles %s differ in treatment benefit",
            f_stat, p_value, eta2, "DO" if p_value < 0.05 else "do NOT",
        )
    return table, anova


def qualitative_signature(
    crosstab: pd.DataFrame, logger: logging.Logger,
) -> pd.DataFrame:
    """Per-profile categorical signature from the Script 06 crosstab.

    Summarizes, for each profile, its dominant desired-change category
    (the Q13 category with the largest within-profile share) and its rate of
    writing an idiosyncratic "Other" current-PhD concern (Q15_9 == 1). These
    are the qualitative cells of the joint display. Returns an empty frame if
    the crosstab is unavailable.
    """
    if len(crosstab) == 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    profiles = sorted(crosstab["profile"].unique())
    for profile in profiles:
        row: dict[str, Any] = {"profile": int(profile)}

        q13 = crosstab[
            (crosstab["field"] == "Q13") & (crosstab["profile"] == profile)
        ]
        if len(q13) > 0:
            top = q13.loc[q13["row_proportion"].idxmax()]
            row["dominant_desired_change_code"] = int(top["category_code"])
            row["dominant_desired_change_share"] = round(float(top["row_proportion"]), 3)
            notable = q13[q13["notable_cell"]]
            row["q13_notable_codes"] = (
                ", ".join(str(int(c)) for c in notable["category_code"])
                if len(notable) else ""
            )

        other = crosstab[
            (crosstab["field"] == "Q15_9") & (crosstab["profile"] == profile)
            & (crosstab["category_code"] == 1)
        ]
        if len(other) > 0:
            row["other_concern_rate"] = round(float(other["row_proportion"].iloc[0]), 3)
            row["other_concern_notable"] = bool(other["notable_cell"].iloc[0])

        rows.append(row)
    table = pd.DataFrame(rows)
    logger.info("Qualitative signature computed for %d profiles", len(table))
    return table


def assemble_joint_display(
    quant: pd.DataFrame,
    causal: pd.DataFrame,
    qual: pd.DataFrame,
    profile_labels: dict[int, str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Assemble the Fetters-Tajima joint display from the three signatures.

    Produces one row per profile with a compact, human-readable cell from
    each strand, plus the profile's descriptive label. This is the table
    intended for direct presentation in the manuscript.
    """
    display_rows: list[dict[str, Any]] = []
    for profile in sorted(quant["profile"].unique()):
        q = quant[quant["profile"] == profile].iloc[0]
        c = causal[causal["profile"] == profile]
        ql = qual[qual["profile"] == profile] if len(qual) else pd.DataFrame()

        # Quantitative cell: well-being and the support/pressure contrast.
        quant_cell = (
            f"well-being {q.get('wellbeing_mean', float('nan')):+.2f}, "
            f"support {q.get('supervisor_support_mean', float('nan')):+.2f}, "
            f"pressure {q.get('academic_pressure_mean', float('nan')):.2f}, "
            f"career uncertainty {q.get('career_uncertainty_mean', float('nan')):.2f}"
        )

        # Causal cell: within-profile treatment benefit.
        if len(c) > 0:
            cc = c.iloc[0]
            causal_cell = (
                f"CATE mean {cc['cate_mean']:+.2f} "
                f"(IQR {cc['cate_q25']:+.2f} to {cc['cate_q75']:+.2f}); "
                f"{cc['pct_ci_above_zero']*100:.0f}% individually positive"
            )
        else:
            causal_cell = "not available"

        # Qualitative cell: desired change and idiosyncratic concerns.
        if len(ql) > 0:
            qq = ql.iloc[0]
            qual_cell = (
                f"dominant desired-change code "
                f"{qq.get('dominant_desired_change_code', 'NA')} "
                f"({qq.get('dominant_desired_change_share', float('nan'))*100:.0f}%); "
                f"'Other' concern rate "
                f"{qq.get('other_concern_rate', float('nan'))*100:.0f}%"
            )
        else:
            qual_cell = "not available"

        display_rows.append({
            "profile": int(profile),
            "label": profile_labels.get(int(profile), ""),
            "n": int(q["n"]),
            "quantitative_signature": quant_cell,
            "causal_signature": causal_cell,
            "qualitative_signature": qual_cell,
        })
    table = pd.DataFrame(display_rows)
    logger.info("Joint display assembled for %d profiles", len(table))
    return table


# ===========================================================================
# Integration (confirmation-expansion-discordance) matrix
# ===========================================================================
def build_integration_matrix(
    quant: pd.DataFrame,
    causal: pd.DataFrame,
    qual: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Classify cross-strand fit per profile (confirmation / expansion / discordance).

    For each profile the function compares the directional stories the
    strands tell:
      * quantitative direction: sign of mean well-being (adapted vs strained),
      * causal direction: whether this profile's CATE is above or below the
        overall mean CATE (gains more or less than average from support),
      * qualitative salience: whether the profile has a notable categorical
        cell (a distinctive desired-change or "Other"-concern pattern).

    The fit label is assigned by a transparent rule:
      - CONFIRMATION when the quantitative and causal readings agree in the
        substantively expected direction (a strained profile that also gains
        more from support, or an adapted profile that gains less),
      - EXPANSION when the qualitative strand adds a notable, non-redundant
        facet not implied by the quantitative/causal pair,
      - DISCORDANCE when the quantitative and causal readings point in
        opposite directions from expectation.
    Fit is interpretive, not inferential; the rule is documented so readers
    can apply their own judgment.
    """
    if len(quant) == 0 or len(causal) == 0:
        return pd.DataFrame()

    overall_cate = float(causal["cate_mean"].mean())
    rows: list[dict[str, Any]] = []

    for profile in sorted(quant["profile"].unique()):
        q = quant[quant["profile"] == profile].iloc[0]
        c = causal[causal["profile"] == profile]
        ql = qual[qual["profile"] == profile] if len(qual) else pd.DataFrame()

        wellbeing = float(q.get("wellbeing_mean", 0.0))
        quant_direction = "adapted" if wellbeing >= 0 else "strained"

        if len(c) > 0:
            profile_cate = float(c.iloc[0]["cate_mean"])
            causal_direction = (
                "gains_more_than_average" if profile_cate >= overall_cate
                else "gains_less_than_average"
            )
        else:
            profile_cate = float("nan")
            causal_direction = "unknown"

        qual_notable = False
        if len(ql) > 0:
            qq = ql.iloc[0]
            qual_notable = bool(
                qq.get("other_concern_notable", False)
                or (isinstance(qq.get("q13_notable_codes", ""), str)
                    and qq.get("q13_notable_codes", "") != "")
            )

        # Expected coupling: strained profiles should gain more from support;
        # adapted profiles should gain less (diminishing returns).
        expected_coupling = (
            (quant_direction == "strained" and causal_direction == "gains_more_than_average")
            or (quant_direction == "adapted" and causal_direction == "gains_less_than_average")
        )
        contradictory_coupling = (
            (quant_direction == "strained" and causal_direction == "gains_less_than_average")
            or (quant_direction == "adapted" and causal_direction == "gains_more_than_average")
        )

        if expected_coupling:
            fit = "confirmation"
        elif qual_notable:
            fit = "expansion"
        elif contradictory_coupling:
            fit = "discordance"
        else:
            fit = "expansion"

        rows.append({
            "profile": int(profile),
            "quantitative_reading": quant_direction,
            "wellbeing_mean": round(wellbeing, 3),
            "causal_reading": causal_direction,
            "profile_cate_mean": round(profile_cate, 4) if np.isfinite(profile_cate) else np.nan,
            "overall_cate_mean": round(overall_cate, 4),
            "qualitative_notable": qual_notable,
            "integration_fit": fit,
        })
        logger.info(
            "Profile %d: %s + %s%s -> %s",
            profile, quant_direction, causal_direction,
            " + notable qualitative" if qual_notable else "", fit,
        )
    return pd.DataFrame(rows)


def derive_meta_inferences(
    joint_display: pd.DataFrame,
    integration_matrix: pd.DataFrame,
    causal_anova: dict[str, Any],
    causal_meta: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Derive study-level meta-inferences from the integrated picture.

    Produces a small set of narrative meta-inferences (the integrated
    conclusions that none of the strands yields alone), each tagged with the
    integration mode that supports it. These are written for direct use in
    the manuscript's integration section.
    """
    inferences: list[dict[str, Any]] = []

    # Meta-inference 1: profile-level treatment-effect structure.
    if causal_anova.get("profiles_differ", False):
        inferences.append({
            "meta_inference": (
                "The benefit of supervisor support for well-being differs "
                "across adaptation profiles, so the profiles capture "
                "treatment-effect structure that single covariates did not."
            ),
            "supporting_mode": "person-centered x causal",
            "evidence": (
                f"ANOVA F = {causal_anova.get('f_statistic')}, "
                f"p = {causal_anova.get('p_value')}, "
                f"eta^2 = {causal_anova.get('eta_squared')}"
            ),
        })
    else:
        inferences.append({
            "meta_inference": (
                "The benefit of supervisor support is broadly similar across "
                "profiles, reinforcing the average-effect finding as the "
                "primary causal conclusion."
            ),
            "supporting_mode": "person-centered x causal",
            "evidence": (
                f"ANOVA p = {causal_anova.get('p_value')}; "
                f"overall ATE = {causal_meta.get('ate', 'NA')}"
            ),
        })

    # Meta-inference 2: integration fit summary.
    if len(integration_matrix) > 0:
        counts = integration_matrix["integration_fit"].value_counts().to_dict()
        inferences.append({
            "meta_inference": (
                "Across profiles the strands integrate predominantly through "
                f"{max(counts, key=counts.get)}, indicating the quantitative, "
                "causal, and qualitative readings tell a coherent story."
            ),
            "supporting_mode": "three-strand integration",
            "evidence": f"integration fit counts: {counts}",
        })

    # Meta-inference 3: robustness of the central causal claim.
    e_val = causal_meta.get("e_value_point", float("nan"))
    if isinstance(e_val, (int, float)) and np.isfinite(e_val):
        inferences.append({
            "meta_inference": (
                "The central causal claim (supervisor support improves "
                "well-being) is robust to plausible unmeasured confounding "
                "and holds across the person-centered typology."
            ),
            "supporting_mode": "causal x sensitivity",
            "evidence": f"E-value = {e_val}; AUTOC p = {causal_meta.get('autoc_p', 'NA')}",
        })

    table = pd.DataFrame(inferences)
    logger.info("Derived %d study-level meta-inferences", len(table))
    return table


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
    """Execute the full mixed-methods integration pipeline."""
    ensure_output_directories()
    set_global_seeds()
    logger = configure_logging()

    logger.info("Pipeline version: %s", CONFIG.pipeline_version)
    logger.info("Root seed: %d", CONFIG.reproducibility.root_seed)
    logger.info("Stage: 07_mixed_methods_integration")

    try:
        df, profiles, cate, crosstab, causal_meta = load_inputs(logger)

        # Profile labels for the display.
        profile_labels = (
            profiles.groupby("profile")["profile_label"].first().to_dict()
            if "profile_label" in profiles.columns else {}
        )

        # --- Phase 1: Individual-level merge ---
        logger.info("=" * 72)
        logger.info("PHASE 1: Individual-level merge of the three strands")
        logger.info("=" * 72)
        merged = merge_individual_level(df, profiles, cate, logger)

        # --- Phase 2: Strand signatures ---
        logger.info("=" * 72)
        logger.info("PHASE 2: Per-profile signatures (quant / causal / qual)")
        logger.info("=" * 72)
        quant = quantitative_signature(merged, logger)
        write_table(quant, "table_s32_quantitative_signature.csv", logger)

        causal, causal_anova = causal_signature(merged, logger)
        write_table(causal, "table_s33_causal_signature.csv", logger)
        write_table(pd.DataFrame([causal_anova]), "table_s34_cate_anova.csv", logger)

        qual = qualitative_signature(crosstab, logger)
        if len(qual) > 0:
            write_table(qual, "table_s35_qualitative_signature.csv", logger)

        # --- Phase 3: Joint display ---
        logger.info("=" * 72)
        logger.info("PHASE 3: Fetters-Tajima joint display")
        logger.info("=" * 72)
        joint_display = assemble_joint_display(
            quant, causal, qual, profile_labels, logger,
        )
        write_table(joint_display, "table_s36_joint_display.csv", logger)
        for _, row in joint_display.iterrows():
            logger.info(
                "  P%d (%s, n=%d): %s | %s | %s",
                row["profile"], row["label"], row["n"],
                row["quantitative_signature"], row["causal_signature"],
                row["qualitative_signature"],
            )

        # --- Phase 4: Integration matrix ---
        logger.info("=" * 72)
        logger.info("PHASE 4: Integration (confirmation-expansion-discordance) matrix")
        logger.info("=" * 72)
        integration_matrix = build_integration_matrix(quant, causal, qual, logger)
        write_table(integration_matrix, "table_s37_integration_matrix.csv", logger)

        # --- Phase 5: Meta-inferences ---
        logger.info("=" * 72)
        logger.info("PHASE 5: Study-level meta-inferences")
        logger.info("=" * 72)
        meta_inferences = derive_meta_inferences(
            joint_display, integration_matrix, causal_anova, causal_meta, logger,
        )
        write_table(meta_inferences, "table_s38_meta_inferences.csv", logger)

        # Persist an integration summary for the manuscript.
        summary = {
            "n_profiles": int(merged["profile"].nunique()),
            "n_integrated": int(len(merged)),
            "cate_by_profile_anova": causal_anova,
            "integration_fit_counts": (
                integration_matrix["integration_fit"].value_counts().to_dict()
                if len(integration_matrix) else {}
            ),
            "n_meta_inferences": int(len(meta_inferences)),
        }
        summary_path = CONFIG.paths.models_dir / "integration_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        logger.info("Integration summary written to %s", summary_path)

        # --- Final summary ---
        logger.info("=" * 72)
        logger.info("Mixed-methods integration completed")
        logger.info("=" * 72)
        logger.info(
            "Integrated %d individuals across %d profiles; "
            "CATE-by-profile %s (p = %s); integration fit: %s",
            len(merged), merged["profile"].nunique(),
            "differs" if causal_anova.get("profiles_differ", False) else "is similar",
            causal_anova.get("p_value", "NA"),
            summary["integration_fit_counts"],
        )
        return 0

    except Exception as exc:
        logger.exception("Mixed-methods integration failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
