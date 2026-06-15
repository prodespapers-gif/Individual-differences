"""
configs.py
==========

Central configuration module for the analysis pipeline supporting the
manuscript "Individual Differences in Doctoral Learning Adaptation and
Well-Being: Academic Pressure, Supervisor Support, Career Uncertainty, and
the Moderating Role of Generative AI among Chinese PhD Students."

This module is the single source of truth for every path, seed, item set,
scoring rule, and analytical threshold used in the pipeline. Every
downstream script imports its configuration from here rather than
hard-coding values, so that any reviewer or replicator can reproduce every
numerical claim in the manuscript by running the pipeline against this
fixed configuration.

Design principles
-----------------
1. Immutability. All configuration objects are frozen dataclasses, which
   prevents accidental in-place modification during analysis. Any script
   that needs to vary a parameter for a sensitivity analysis must do so
   explicitly through a function argument rather than by mutating the
   shared configuration object.
2. Determinism. A single root seed governs every stochastic procedure.
   Downstream stages derive their own sub-seeds by adding stage-specific
   offsets, so independent stages use distinct but reproducible random
   streams.
3. Fidelity to the data as published. Every item set and scoring rule
   below was verified directly against the anonymised public Nature
   Careers Graduate Survey 2025 dataset (the "Codes" worksheet) rather
   than assumed from item labels. Three properties of that dataset shape
   the entire pipeline and are documented at the point of use:

     (a) The Q15 ("concerns about your PhD") and Q16 ("concerns about life
         after your PhD") blocks are BINARY multiple-select checkboxes
         coded 0/1, not Likert ratings. Academic pressure and career
         uncertainty are therefore scored as the COUNT (equivalently, the
         proportion) of endorsed concerns within each block, and are NOT
         submitted to reverse-coding, internal-consistency reliability of
         the Likert kind, or single-factor CFA. They are formative
         indices, not reflective scales.

     (b) The Likert blocks do not share a common maximum. Q12 is on a
         1-5 metric, Q20/Q33/Q34 on 1-6, Q11_New on 1-7, and the Q14a
         satisfaction block on 1-8. Reverse-coding is therefore performed
         per item against that item's own scale maximum, never against a
         single global constant.

     (c) On the Chinese subset (N = 400) there is no item-level
         missingness on any analysis variable. Multiple imputation is
         retained as an OPTIONAL, missingness-aware safeguard so the
         pipeline remains correct if applied to a future wave with
         missing data, but on the present data it is a no-op by design
         and the analyses run on the single observed dataset. This is
         stated transparently rather than running an imputation engine
         that imputes nothing.

4. Transparency. Every analytical choice that could affect a reported
   number is recorded here with an inline rationale.

Methodological references underlying the analytical choices
-----------------------------------------------------------
Bollen, K. A., & Lennox, R. (1991). Conventional wisdom on measurement:
    A structural equation perspective. Psychological Bulletin, 110,
    305-314. [Formative vs. reflective indicators; governs the treatment
    of the Q15/Q16 concern checklists.]
Chen, F. F. (2007). Sensitivity of goodness of fit indexes to lack of
    measurement invariance. Structural Equation Modeling, 14, 464-504.
Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in
    covariance structure analysis. Structural Equation Modeling, 6, 1-55.
Nylund, K. L., Asparouhov, T., & Muthen, B. O. (2007). Deciding on the
    number of classes in latent class analysis. Structural Equation
    Modeling, 14, 535-569.
Spurk, D., Hirschi, A., Wang, M., Valero, D., & Kauffeld, S. (2020).
    Latent profile analysis: A review and how-to guide of its application
    within vocational behavior research. Journal of Vocational Behavior,
    120, 103445.
Sverdrup, E., Petukhova, M., & Wager, S. (2025). Estimating treatment
    effect heterogeneity in psychiatry: A review and tutorial with causal
    forests. International Journal of Methods in Psychiatric Research,
    34(2), e70015.
van Buuren, S. (2018). Flexible imputation of missing data (2nd ed.).
    Chapman and Hall/CRC.

Usage
-----
    from configs import CONFIG, ensure_output_directories, set_global_seeds

    ensure_output_directories()
    set_global_seeds()
    df = pd.read_csv(CONFIG.paths.chinese_phd_dataset)


Target journal: Learning and Individual Differences (Elsevier)
Python version: 3.10 or later
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# The project root is resolved relative to this file's location, which makes
# the codebase portable across machines without environment-variable
# configuration. This file lives in <project_root>/src/configs.py, so the
# project root is the parent of this file's directory.

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Dependency version constants (pinned for reproducibility)
# ---------------------------------------------------------------------------
# These are the version ranges against which the pipeline was developed and
# tested. The accompanying requirements.txt should pin exact versions. Any
# environment satisfying these constraints should reproduce identical
# numerical results given the same root seed.

PYTHON_MIN_VERSION: Final[tuple[int, int, int]] = (3, 10, 0)

REQUIRED_PACKAGE_VERSIONS: Final[dict[str, str]] = {
    "numpy": ">=1.26,<2.2",
    "pandas": ">=2.1,<2.3",
    "scipy": ">=1.11,<2.0",
    "scikit-learn": ">=1.4,<1.6",
    "statsmodels": ">=0.14,<0.16",
    "semopy": ">=2.3,<3.0",
    "xgboost": ">=2.0,<3.0",
    "shap": ">=0.44,<0.46",
    "econml": ">=0.15,<0.16",
    "bertopic": ">=0.16,<0.17",
    "sentence-transformers": ">=2.5,<4.0",
    "umap-learn": ">=0.5,<0.6",
    "hdbscan": ">=0.8.33,<0.9",
    "joblib": ">=1.3,<2.0",
    "openpyxl": ">=3.1,<4.0",
}


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths for inputs, intermediate artifacts, and outputs.

    All paths are absolute and resolved relative to the project root, which
    eliminates a common class of bugs that arise when scripts are executed
    from different working directories.

    A helper is provided for optional imputation-indexed dataset paths.
    Because the published Chinese subset has no item-level missingness,
    imputation is off by default (see ImputationConfig); when it is enabled
    for a sensitivity analysis or a future wave, these paths name the
    completed datasets.
    """

    # --- Raw input (user-supplied; never modified by the pipeline) ---
    # The anonymised public Nature Careers Graduate Survey 2025 workbook.
    # Only the "Codes" worksheet is authoritative; the "Labels" worksheet
    # headers are offset by one column in the public release and must not
    # be used for code-to-meaning alignment (see codebook construction in
    # 01_data_preparation.py).
    raw_nature_xlsx: Path = PROJECT_ROOT / "data" / "raw" / (
        "Nature Graduate Survey_Raw_Data_anonymised for publishing.xlsx"
    )
    raw_nature_sheet: str = "Codes"

    # --- Processed analysis dataset (produced by Script 01) ---
    # This is the canonical, scored, analysis-ready dataset that every
    # downstream script consumes.
    chinese_phd_dataset: Path = PROJECT_ROOT / "data" / "chinese_phd_dataset.csv"

    # Codebook documenting every derived variable and scoring rule.
    codebook: Path = PROJECT_ROOT / "data" / "codebook.json"

    # Optional multiple-imputation outputs (only written when imputation is
    # enabled; see ImputationConfig.enable). Indexed completed datasets and
    # a diagnostics file live alongside the canonical dataset.
    processed_data_dir: Path = PROJECT_ROOT / "data"
    imputation_diagnostics: Path = (
        PROJECT_ROOT / "data" / "imputation_diagnostics.csv"
    )

    # --- Output directories ---
    output_root: Path = PROJECT_ROOT / "outputs"
    tables_dir: Path = PROJECT_ROOT / "outputs" / "tables"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    models_dir: Path = PROJECT_ROOT / "outputs" / "models"
    logs_dir: Path = PROJECT_ROOT / "outputs" / "logs"

    def imputed_path(self, imputation_index: int) -> Path:
        """Construct the path for one optional imputed dataset.

        Parameters
        ----------
        imputation_index : int
            Imputation number from 1 to ``ImputationConfig.n_imputations``.

        Returns
        -------
        Path
            Absolute path to the completed CSV for that imputation.
        """
        return (
            self.processed_data_dir
            / f"chinese_phd_dataset_imp{imputation_index:02d}.csv"
        )

    def all_imputed_paths(self, n_imputations: int) -> list[Path]:
        """List all optional imputed dataset paths."""
        return [self.imputed_path(i) for i in range(1, n_imputations + 1)]


# ---------------------------------------------------------------------------
# Reproducibility configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReproducibilityConfig:
    """Random seeds and determinism controls.

    A single root seed governs all stochastic procedures. Downstream stages
    derive sub-seeds by adding stage-specific offsets, which ensures that
    independent stages (for example, the LPA bootstrap and the causal
    forest) use distinct but reproducible random streams.

    The root seed is the project initiation date in YYYYMMDD format, a
    convention that makes the seed self-documenting.
    """

    root_seed: int = 20260607  # Locked at preregistration; do not modify.

    # Stage-specific seed offsets.
    imputation_seed_offset: int = 500
    psychometric_seed_offset: int = 750
    lpa_bootstrap_seed_offset: int = 1000
    predictive_seed_offset: int = 3000
    causal_forest_seed_offset: int = 2000
    bertopic_seed_offset: int = 4000
    integration_seed_offset: int = 5000

    # Deterministic mode for any GPU operations (PyTorch / Hugging Face).
    cuda_deterministic: bool = True
    cudnn_benchmark: bool = False  # Sacrifices speed for determinism.


# ---------------------------------------------------------------------------
# Study configuration: Nature Careers Graduate Survey 2025 (Chinese subset)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StudyConfig:
    """Operationalization of every construct for the Chinese PhD subset.

    The country code 36 corresponds to China in the Nature dataset's
    numeric scheme. The dual inclusion criterion (Q7 == 36 OR Q6a == 36)
    yields exactly N = 400 respondents, verified against the published
    Codes worksheet: 52 currently studying in China (Q7 == 36), 352 of
    Chinese origin (Q6a == 36), of whom 348 study abroad, for 400 unique
    respondents under the union. The official Nature Careers report uses
    n = 312 for its China figures, but the derivation of that subset is
    not documented in the public file; the n = 400 union is adopted here
    because it is exactly reproducible from the openly available data, and
    the distinction between "studying in China" and "Chinese origin" is
    preserved as an audit covariate (see 01_data_preparation.py) so that
    a reviewer can inspect either subpopulation.
    """

    # --- Inclusion criteria ---
    china_country_code: int = 36
    inclusion_rule: str = "Q7 == 36 OR Q6a == 36"
    expected_sample_size: int = 400

    # =====================================================================
    # CONSTRUCT 1: Academic pressure  --  FORMATIVE BINARY INDEX
    # =====================================================================
    # The Q15 block ("Overall, what concerns you most about your PhD
    # experience at the moment, if anything?") is a multiple-select
    # checklist: each item is coded 1 if the respondent endorsed that
    # concern and 0 otherwise. These are formative indicators of a stressor
    # load, not reflective indicators of a latent trait (Bollen & Lennox,
    # 1991). Academic pressure is therefore the COUNT of endorsed concerns
    # among the items below (range 0-6), reported alongside the proportion.
    # The items are NOT reverse-coded, NOT submitted to Cronbach's alpha or
    # single-factor CFA, and NOT treated as a continuous Likert mean.
    academic_pressure_items: tuple[str, ...] = (
        "Q15_1",  # Lack of available funding
        "Q15_2",  # Lack of adequate mentoring from supervisor
        "Q15_3",  # Financial pressures
        "Q15_4",  # Mental health
        "Q15_6",  # Pressure to get published
        "Q15_7",  # Imposter syndrome
    )
    # Q15_5 (family/caring responsibilities) and Q15_8 (political landscape)
    # tap distinct concerns and are excluded from the index but retained as
    # auxiliary binary variables. Q15_9 is a free-text "Other" field handled
    # by the qualitative pipeline (Script 06).

    # =====================================================================
    # CONSTRUCT 2: Supervisor support  --  REFLECTIVE LIKERT SCALE
    # =====================================================================
    # Two Likert sources combine into supervisor support. The Q20 block
    # ("...regarding your supervisor") is agree-disagree on a 1-6 metric
    # where higher = more agreement that the supervisor is supportive
    # (these are NOT reverse-keyed). The two Q14a satisfaction items are on
    # a 1-8 metric whose anchors run from satisfied to dissatisfied in a
    # direction that requires reverse-coding so that higher = more
    # satisfaction. Reverse-coding is applied per item against that item's
    # own maximum (see items_to_reverse and likert_scale_max below).
    supervisor_support_items: tuple[str, ...] = (
        "Q20_1",   # Makes time for frank career conversations  (1-6)
        "Q20_2",   # Open to non-academic careers               (1-6)
        "Q20_3",   # Useful advice for non-academic careers     (1-6)
        "Q20_4",   # Encouraged attendance at career events     (1-6)
        "Q14a_3",  # Satisfaction: overall relationship w/ supervisor (1-8, reversed)
        "Q14a_5",  # Satisfaction: guidance received about research  (1-8, reversed)
    )

    # =====================================================================
    # CONSTRUCT 3: Career uncertainty  --  FORMATIVE BINARY INDEX
    # =====================================================================
    # The Q16 block ("Thinking about life after your PhD, what concerns you
    # most, if anything?") is, like Q15, a 0/1 multiple-select checklist.
    # Career uncertainty is the COUNT of endorsed post-PhD concerns among
    # the items below (range 0-7), reported alongside the proportion. Same
    # formative treatment as academic pressure: no reverse-coding, no
    # internal-consistency reliability, no single-factor CFA.
    career_uncertainty_items: tuple[str, ...] = (
        "Q16_1",  # Not feeling qualified for next career move
        "Q16_2",  # Lack of guidance for next career move
        "Q16_3",  # Lack of available research jobs
        "Q16_4",  # Only being offered temporary contracts
        "Q16_6",  # Financial pressures after graduation
        "Q16_8",  # Burnout / mental health concerns
        "Q16_9",  # Not feeling professionally fulfilled
    )
    # Q16_5 (overall value of the degree), Q16_7 (work-life balance),
    # Q16_10 (Other free text), and Q16_11 (None of the above) are excluded
    # from the index. Q16_11 is retained as an explicit "no concerns" flag.

    # =====================================================================
    # CONSTRUCT 4: Well-being  --  REFLECTIVE LIKERT SCALE
    # =====================================================================
    # Overall PhD satisfaction and three agree-disagree experience items.
    # Q11_New is on a 1-7 metric (higher = more satisfied) and the three
    # Q12 items are on a 1-5 metric (higher = more agreement). All four are
    # already keyed so that higher = better well-being, so none is reversed.
    # The composite is the mean of per-item z-scores (computed in Script 01)
    # rather than a raw mean, because the items live on different metrics.
    wellbeing_items: tuple[str, ...] = (
        "Q11_New",  # Overall PhD satisfaction        (1-7)
        "Q12_1",    # Enjoying PhD experience overall (1-5)
        "Q12_2",    # Feeling fulfilled by the work   (1-5)
        "Q12_3",    # Reality aligns with expectations(1-5)
    )

    # =====================================================================
    # MODERATORS: Generative AI use, comfort, concerns  (novel contribution)
    # =====================================================================
    # Q32 is the AI-use item. In the public file it is a 1-6 response about
    # comfort/use rather than a clean yes/no, so it is treated as an ordinal
    # AI-engagement indicator and additionally dichotomized in Script 01
    # (see ai_use_positive_codes) for the binary-moderator analyses. The
    # Q33 block (7 items, 1-6) measures comfort with AI tools; the Q34 block
    # (9 items, 1-6) measures agreement with statements about AI in research
    # (concerns/attitudes). Both Q33 and Q34 are reflective Likert scales
    # scored as item means after any necessary reverse-coding (none of the
    # Q33/Q34 items requires reversal under the published keying).
    ai_use_item: str = "Q32"
    ai_use_positive_codes: tuple[int, ...] = (4, 5)  # "comfortable" end -> AI user
    ai_comfort_items: tuple[str, ...] = (
        "Q33_1", "Q33_2", "Q33_3", "Q33_4",
        "Q33_5", "Q33_6", "Q33_7",
    )
    ai_concern_items: tuple[str, ...] = (
        "Q34_1", "Q34_2", "Q34_3", "Q34_4", "Q34_5",
        "Q34_6", "Q34_7", "Q34_8", "Q34_9",
    )

    # =====================================================================
    # Open-response (qualitative) fields, handled by Script 06
    # =====================================================================
    open_response_items: tuple[str, ...] = (
        "Q13",     # One thing you would change about the program
        "Q15_9",   # Other current PhD concerns (free text)
        "Q16_10",  # Other post-PhD concerns (free text)
    )

    # =====================================================================
    # Demographic / control variables retained in the analysis frame
    # =====================================================================
    # NOTE ON SEMANTICS: in the anonymised public file the demographic codes
    # do not all carry the textbook meaning suggested by the variable names,
    # because the Labels worksheet is column-shifted. The fields below are
    # retained as ordinal/categorical controls under their CODE values, and
    # 01_data_preparation.py documents the verified value ranges in the
    # codebook. Q3a (PhD stage, 1-9) and Q3b (full/part-time, 1-2) are the
    # only demographics whose code semantics are unambiguous in the Codes
    # worksheet; the remainder are used as adjustment covariates without a
    # substantive label claim.
    demographic_items: tuple[str, ...] = (
        "Q3a",  # PhD stage (1 = <1 year ... 9 = >7 years) -- verified ordinal
        "Q3b",  # Full-time (1) / part-time (2)            -- verified binary
        "Q17",  # Weekly hours on PhD (ordinal bands)      -- control
        "Q18",  # Supervisor contact (ordinal bands)       -- control
        "Q36",  # Retained as categorical control (code values; see codebook)
        "Q37",  # Retained as categorical control (code values; see codebook)
        "Q38",  # Retained as categorical control (code values; see codebook)
    )

    # Country variables used in the inclusion logic, retained for audit.
    country_items: tuple[str, ...] = ("Q6a", "Q7")

    # =====================================================================
    # Reverse-coding registry: item -> that item's own scale maximum
    # =====================================================================
    # Reverse-coding maps x -> (scale_max + 1 - x), applied PER ITEM against
    # the item's own maximum. No global likert_max constant exists, because
    # the Likert blocks span 1-5, 1-6, 1-7, and 1-8.
    #
    # VERIFIED EMPTY. The anonymised public release has already normalized
    # item directions: every reflective item is keyed so that higher =
    # more of the construct. This was confirmed directly against the data
    # in two ways. (i) The four Q20 supervisor-agreement items and the two
    # Q14a supervisor-satisfaction items intercorrelate POSITIVELY once the
    # metric difference (1-6 vs 1-8) is removed by item-standardization
    # (full-scale standardized alpha = 0.87); reverse-coding the Q14a items
    # would flip them against Q20 and collapse the scale to alpha = 0.18.
    # (ii) Both Q20 and raw Q14a correlate POSITIVELY with overall PhD
    # satisfaction (Q11_New), confirming a common "higher = more support"
    # direction on substantive grounds. The Q15/Q16 binary blocks are
    # formative 0/1 indicators and are never reversed in any case. If a
    # future, non-anonymised wave restores the original (un-normalized)
    # item directions, populate this tuple accordingly.
    items_to_reverse: tuple[str, ...] = ()

    likert_scale_max: dict[str, int] = field(
        default_factory=lambda: {
            "Q11_New": 7,
            "Q12_1": 5, "Q12_2": 5, "Q12_3": 5,
            "Q14a_3": 8, "Q14a_5": 8,
            "Q20_1": 6, "Q20_2": 6, "Q20_3": 6, "Q20_4": 6,
            "Q32": 6,
            "Q33_1": 6, "Q33_2": 6, "Q33_3": 6, "Q33_4": 6,
            "Q33_5": 6, "Q33_6": 6, "Q33_7": 6,
            "Q34_1": 6, "Q34_2": 6, "Q34_3": 6, "Q34_4": 6, "Q34_5": 6,
            "Q34_6": 6, "Q34_7": 6, "Q34_8": 6, "Q34_9": 6,
        }
    )

    # Registry of which constructs are formative (binary count indices) vs.
    # reflective (Likert scales). Downstream scripts consult this so that
    # reliability and CFA are run ONLY on reflective scales.
    formative_constructs: tuple[str, ...] = (
        "academic_pressure", "career_uncertainty",
    )
    reflective_constructs: tuple[str, ...] = (
        "supervisor_support", "wellbeing", "ai_comfort", "ai_concerns",
    )


# ---------------------------------------------------------------------------
# Optional multiple-imputation configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImputationConfig:
    """Configuration for optional multiple imputation by chained equations.

    IMPORTANT: On the published Chinese subset (N = 400) there is no
    item-level missingness on any analysis variable, verified directly
    against the Codes worksheet. Imputation is therefore DISABLED by
    default (``enable = False``) and the pipeline runs on the single
    observed dataset. The machinery is retained, fully functional, so the
    pipeline remains correct and reproducible if applied to a future wave
    or a sensitivity analysis that does contain missing data; in that case
    set ``enable = True``.

    When enabled, multiple imputation rather than single imputation is used
    so that downstream confidence intervals incorporate imputation
    uncertainty via Rubin's rules. The number of imputations follows van
    Buuren's (2018) rule of thumb that M should at least equal the
    percentage of incomplete cases; M = 20 is conservative for the low
    missingness rates anticipated in any future wave.
    """

    enable: bool = False  # No missingness in the published data; off by design.

    n_imputations: int = 20
    max_iter_per_imputation: int = 50
    sample_posterior: bool = True
    tol: float = 1e-3
    estimator_name: str = "BayesianRidge"

    # Listwise-deletion thresholds applied before imputation (only relevant
    # when enable=True; with no missingness they never remove a case).
    case_missing_threshold: float = 0.50   # Drop cases > 50% missing on focal items
    item_missing_threshold: float = 0.40   # Drop items > 40% missing across cases

    compute_fmi: bool = True  # Fraction-of-missing-information diagnostics.


# ---------------------------------------------------------------------------
# Psychometric validation configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PsychometricConfig:
    """Configuration for measurement validation (reflective scales only).

    Reliability and confirmatory factor analysis are computed ONLY for the
    reflective Likert scales named in StudyConfig.reflective_constructs.
    The formative binary indices (academic pressure, career uncertainty)
    are summarized by endorsement frequencies and inter-item tetrachoric
    structure for description only; they are never assigned a Cronbach's
    alpha or a single-factor CFA, because internal-consistency reliability
    is undefined for formative indicators (Bollen & Lennox, 1991).
    """

    # Reliability thresholds (Nunnally & Bernstein, 1994).
    alpha_acceptable: float = 0.70
    omega_acceptable: float = 0.70

    # CFA fit-index thresholds (Hu & Bentler, 1999).
    cfa_cfi_threshold: float = 0.95
    cfa_tli_threshold: float = 0.95
    cfa_rmsea_threshold: float = 0.06
    cfa_srmr_threshold: float = 0.08

    # Measurement-invariance change-in-fit thresholds (Cheung & Rensvold,
    # 2002; Chen, 2007). Both delta-CFI and delta-RMSEA must be satisfied;
    # delta-SRMR is reported as a supplementary index. Invariance is tested
    # across the verified binary control Q3b (full-/part-time) as the
    # grouping variable, the only demographic with unambiguous two-group
    # semantics in the public file.
    invariance_grouping_variable: str = "Q3b"
    invariance_delta_cfi: float = 0.01
    invariance_delta_rmsea: float = 0.015
    invariance_delta_srmr_metric: float = 0.030
    invariance_delta_srmr_scalar: float = 0.015

    # Partial-invariance fallback (Steenkamp & Baumgartner, 1998): if full
    # invariance fails, iteratively free the worst-fitting constraint until
    # partial invariance holds or the freed proportion exceeds the ceiling.
    enable_partial_invariance: bool = True
    max_proportion_noninvariant: float = 0.50

    # Reliability bootstrap configuration.
    alpha_bootstrap_iterations: int = 1000
    omega_bootstrap_iterations: int = 500

    ci_level: float = 0.95


# ---------------------------------------------------------------------------
# Latent Profile Analysis configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LPAConfig:
    """Configuration for latent profile analysis.

    Hyperparameters follow Spurk et al. (2020) and Nylund et al. (2007).
    Profiles are estimated over the four focal constructs after each is
    placed on a common z-score metric in Script 01: the two formative
    indices (academic pressure, career uncertainty) are standardized
    counts, and the two reflective scales (supervisor support, well-being)
    are standardized scale scores. Mixing standardized formative and
    reflective indicators in a profile model is appropriate because LPA
    operates on the observed indicator vector, not on a latent measurement
    model.
    """

    # Profile-defining features (resolved to standardized columns in Script 01).
    profile_features: tuple[str, ...] = (
        "academic_pressure_z",
        "supervisor_support_z",
        "career_uncertainty_z",
        "wellbeing_z",
    )

    # Model selection.
    min_classes: int = 1
    max_classes: int = 6
    n_random_starts: int = 1000   # Guards against local maxima (Spurk et al., 2020).
    max_em_iterations: int = 5000
    em_convergence_tolerance: float = 1e-6

    # Covariance structures to compare.
    variance_structures: tuple[str, ...] = ("equal", "varying", "varying_cov")

    # Model-selection criteria, in priority order.
    primary_criterion: str = "BIC"
    secondary_criteria: tuple[str, ...] = ("AIC", "SABIC", "entropy")

    # Bootstrap likelihood-ratio test.
    use_blrt: bool = True
    blrt_n_bootstrap: int = 500

    # Profile-stability bootstrap (Adjusted Rand Coefficient).
    stability_n_bootstrap: int = 1000
    stability_min_arc: float = 0.70

    # Minimum profile size (Nylund-Gibson & Choi, 2018).
    min_profile_proportion: float = 0.05

    # Entropy threshold for accepting a solution.
    entropy_acceptable: float = 0.70

    # Validate profile membership with multinomial logistic regression
    # (Tikkanen et al., 2021).
    validate_with_logistic_regression: bool = True


# ---------------------------------------------------------------------------
# Predictive modeling and AI-moderation configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictiveModelingConfig:
    """Configuration for the predictive and AI-moderation analyses.

    Two complementary methods. Classical OLS moderation estimates linear
    interaction effects of each AI moderator on the focal-predictor ->
    well-being relationships, with bootstrap confidence intervals.
    Gradient boosting with SHAP detects non-linear moderation through
    tree ensembles and the SHAP interaction-value decomposition. Because
    imputation is off on the present data, pooling collapses to the single
    observed dataset; the Rubin's-rules machinery remains available for the
    enable=True path.
    """

    # Classical moderation.
    n_bootstrap_ci: int = 10000
    ci_level: float = 0.95
    centering_method: str = "mean"  # Mean-center predictors before product.

    # XGBoost.
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 500
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_early_stopping_rounds: int = 50
    cv_n_folds: int = 10
    cv_n_repeats: int = 5

    # SHAP interaction values.
    shap_n_background_samples: int = 200
    shap_max_evals: int = 10000

    pool_across_imputations: bool = True  # Only active when imputation enabled.


# ---------------------------------------------------------------------------
# Causal forest configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CausalForestConfig:
    """Configuration for individual treatment-effect estimation.

    Follows Sverdrup, Petukhova, and Wager (2025) for behavioral-science
    applications of causal forests. The treatment is dichotomized
    supervisor support; the outcome is well-being; effect modifiers are
    the other focal constructs plus controls. The AUTOC/RATE statistic,
    train-test split with quartile stratification, best linear projection,
    and E-value sensitivity analysis are all included.
    """

    n_trees: int = 5000
    min_node_size: int = 5
    sample_fraction: float = 0.5
    honest_splitting: bool = True
    n_cv_folds: int = 5  # Cross-fitting (Chernozhukov et al., 2018).

    treatment_dichotomization: str = "median"

    # Train-test split for heterogeneity validation (60/40 in small samples;
    # Sverdrup et al., 2025).
    train_proportion: float = 0.60
    stratify_test_by_quartile: bool = True

    calibration_test: bool = True
    calibration_n_quantiles: int = 4

    # RATE / AUTOC (Yadlowsky et al., 2025). Use the rank-weighted
    # average-treatment-effect estimator with bootstrap inference, not an
    # ad hoc cumulative-mean approximation.
    compute_autoc: bool = True
    autoc_n_bootstrap: int = 500

    compute_best_linear_projection: bool = True
    compute_qini_curve: bool = True

    # Rosenbaum bounds and E-value (VanderWeele & Ding, 2017).
    rosenbaum_gamma_range: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0)

    pool_across_imputations: bool = True  # Only active when imputation enabled.


# ---------------------------------------------------------------------------
# Qualitative NLP configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QualitativeNLPConfig:
    """Configuration for the qualitative open-response analyses."""

    sentence_transformer_model: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    embedding_batch_size: int = 32

    # BERTopic / UMAP / HDBSCAN (McInnes et al., 2018; Grootendorst, 2022).
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5

    top_n_words_per_topic: int = 10
    min_topic_size: int = 10
    nr_topics: str = "auto"

    # Inter-rater reliability (Krippendorff, 2019).
    krippendorff_alpha_threshold: float = 0.67
    manual_validation_n: int = 100


# ---------------------------------------------------------------------------
# Mixed-methods integration configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IntegrationConfig:
    """Configuration for mixed-methods integration.

    With a single sample there is no cross-sample profile matching. The
    integration stage instead aligns the quantitative profiles (Script 03)
    with the qualitative themes (Script 06) and the causal heterogeneity
    (Script 05), and assembles the Fetters-Tajima joint display and the
    integration matrix that operationalize the mixed-methods design.
    """

    # Profile-by-CATE integration.
    min_profile_size_for_cate: int = 5

    # Profile-by-theme integration: chi-square with standardized residuals.
    residual_flag_threshold: float = 2.0

    # Joint display (Fetters & Tajima, 2022).
    joint_display_format: str = "statistics_by_themes"
    include_box_plots: bool = True


# ---------------------------------------------------------------------------
# Hardware configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareConfig:
    """Hardware-aware execution parameters.

    Defaults are conservative so the pipeline runs on a workstation or a
    laptop. ``n_cpu_workers = -1`` lets joblib use all available cores;
    set a positive integer to cap parallelism on a shared machine.
    """

    n_cpu_workers: int = -1  # -1 => all cores (joblib convention).
    n_gpu_devices: int = 0   # Set > 0 only if CUDA GPUs are present.
    gpu_memory_fraction: float = 0.85
    use_mixed_precision: bool = True

    joblib_backend: str = "loky"
    # Backend for embarrassingly parallel model-refitting loops (LPA BLRT
    # and stability bootstraps, causal-forest bootstraps). "threading" is
    # the default because scikit-learn's EM and forest fits release the GIL
    # during their compiled inner loops, so threads parallelize them with
    # near-linear speedup while avoiding the process-pool pickling and
    # module-re-import fragility of "loky" when a stage module is launched
    # under a non-standard name. Set to "loky" for pure-Python inner loops.
    parallel_backend: str = "threading"
    joblib_temp_folder: str = "/tmp/joblib"

    pytorch_cuda_alloc_conf: str = "max_split_size_mb:512"


# ---------------------------------------------------------------------------
# Master configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Top-level configuration aggregating all subconfigurations.

    Instantiated once at module load as the singleton ``CONFIG`` object,
    which all downstream scripts import.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )
    study: StudyConfig = field(default_factory=StudyConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    psychometric: PsychometricConfig = field(default_factory=PsychometricConfig)
    lpa: LPAConfig = field(default_factory=LPAConfig)
    predictive: PredictiveModelingConfig = field(
        default_factory=PredictiveModelingConfig
    )
    causal_forest: CausalForestConfig = field(default_factory=CausalForestConfig)
    qualitative: QualitativeNLPConfig = field(default_factory=QualitativeNLPConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    # Manuscript metadata, recorded for traceability.
    manuscript_title: str = (
        "Individual Differences in Doctoral Learning Adaptation and "
        "Well-Being: Academic Pressure, Supervisor Support, Career "
        "Uncertainty, and the Moderating Role of Generative AI among "
        "Chinese PhD Students"
    )
    target_journal: str = "Learning and Individual Differences"
    pipeline_version: str = "2.0.0"


# Singleton instance imported by all downstream scripts.
CONFIG: Final[Config] = Config()


# ---------------------------------------------------------------------------
# Initialization utilities
# ---------------------------------------------------------------------------
def ensure_output_directories() -> None:
    """Create output directories if they do not exist.

    Called once at the start of each analysis script. Idempotent.
    """
    for directory in (
        CONFIG.paths.processed_data_dir,
        CONFIG.paths.output_root,
        CONFIG.paths.tables_dir,
        CONFIG.paths.figures_dir,
        CONFIG.paths.models_dir,
        CONFIG.paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seeds(override_seed: int | None = None) -> int:
    """Set deterministic random seeds across all relevant libraries.

    Covers Python's ``random``, NumPy, the ``PYTHONHASHSEED`` environment
    variable, and PyTorch if it is installed. Called once at the start of
    each analysis script before any stochastic operation.

    Parameters
    ----------
    override_seed : int or None
        If supplied, overrides the configured root seed for this process.
        Used by main.py to support sensitivity analyses; the override does
        not mutate the singleton CONFIG object.

    Returns
    -------
    int
        The seed that was actually applied.
    """
    import os
    import random

    import numpy as np

    seed = (
        override_seed
        if override_seed is not None
        else CONFIG.reproducibility.root_seed
    )
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if CONFIG.reproducibility.cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = (
                CONFIG.reproducibility.cudnn_benchmark
            )
        if torch.cuda.is_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                CONFIG.hardware.pytorch_cuda_alloc_conf
            )
    except ImportError:
        # PyTorch is optional for stages that do not require it.
        pass

    return seed


def derive_imputation_seed(imputation_index: int) -> int:
    """Compute a deterministic seed for one optional imputation.

    Each imputation receives a distinct seed derived from the root seed and
    the imputation index, so any individual imputation can be reproduced
    exactly without rerunning the full pipeline.

    Parameters
    ----------
    imputation_index : int
        Imputation number from 1 to ``ImputationConfig.n_imputations``.

    Returns
    -------
    int
        Seed for the specified imputation.
    """
    return (
        CONFIG.reproducibility.root_seed
        + CONFIG.reproducibility.imputation_seed_offset
        + imputation_index
    )


if __name__ == "__main__":
    # Allow `python configs.py` to verify the configuration is well-formed.
    ensure_output_directories()
    print("Configuration loaded successfully.")
    print(f"Project root:            {PROJECT_ROOT}")
    print(f"Expected sample size:    {CONFIG.study.expected_sample_size}")
    print(f"Root seed:               {CONFIG.reproducibility.root_seed}")
    print(f"Imputation enabled:      {CONFIG.imputation.enable}")
    print(f"Formative constructs:    {CONFIG.study.formative_constructs}")
    print(f"Reflective constructs:   {CONFIG.study.reflective_constructs}")
    print(f"Items to reverse-code:   {CONFIG.study.items_to_reverse}")
    print(f"LPA random starts:       {CONFIG.lpa.n_random_starts}")
    print(f"Causal forest trees:     {CONFIG.causal_forest.n_trees}")
    print(f"Pipeline version:        {CONFIG.pipeline_version}")
