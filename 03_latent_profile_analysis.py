"""
03_latent_profile_analysis.py
=============================

Latent profile analysis pipeline (Stage 3 of 7) for the manuscript
"Individual Differences in Doctoral Learning Adaptation and Well-Being:
Academic Pressure, Supervisor Support, Career Uncertainty, and the
Moderating Role of Generative AI among Chinese PhD Students."

This script identifies distinct subgroups of doctoral students from their
configurations across the four focal constructs (academic pressure,
supervisor support, career uncertainty, well-being). Profiles are the
person-centered backbone of the study: every later stage (causal
heterogeneity, qualitative themes, integration) is interpreted against
them.

The analysis operates on the standardized profile features produced by
Script 01 (academic_pressure_z, supervisor_support_z, career_uncertainty_z,
wellbeing_z). Two of these derive from formative binary indices and two
from reflective Likert scales; mixing them is appropriate because latent
profile analysis models the observed indicator vector, not a latent
measurement model, so the distinction between formative and reflective
measurement that governed Script 02 does not constrain Script 03.

Because Script 01 produces a single analysis dataset (multiple imputation
is disabled: the published data have no missingness), the pipeline fits
one set of models rather than pooling across imputations. The
imputation-pooling machinery of earlier drafts is therefore removed; if a
future wave enables imputation, profiles would be estimated per imputation
and aligned, but that is out of scope for the present data and is not
simulated here.

Pipeline stages
---------------
1. Load the analysis dataset and assemble the standardized feature matrix.
2. Fit Gaussian mixture models for K = 1..6 with many random starts to
   avoid local maxima.
3. Compute BIC, AIC, SABIC, entropy, and minimum profile size for each K.
4. Run the bootstrap likelihood-ratio test (BLRT) for each K vs. K-1.
5. Select the optimal K with the Spurk et al. (2020) hierarchy: among
   admissible solutions (converged, all profiles >= 5%), prefer the BIC
   minimum, cross-checked against the BLRT and an entropy floor.
6. Characterize the chosen solution by standardized construct means and
   profile sizes, and assign provisional descriptive labels.
7. Assess profile stability by nonparametric bootstrap (Adjusted Rand
   Coefficient between the reference solution and each resample).
8. Validate profile membership with multinomial logistic regression on
   demographic and contextual covariates.
9. Write the selection, BLRT, characterization, stability, validation, and
   assignment tables consumed by Scripts 05-07.

Runtime
-------
The model battery, BLRT (CONFIG.lpa.blrt_n_bootstrap replicates per K step),
and stability bootstrap (CONFIG.lpa.stability_n_bootstrap replicates, each a
full refit) are the compute-intensive stages. With the camera-ready budgets
(1000 random starts, 500 BLRT, 1000 stability) the stage runs comfortably on
the target multi-core server; on a small machine, reduce these budgets for a
quick check. Embarrassingly parallel loops use the threading backend
(CONFIG.hardware.parallel_backend) because scikit-learn's EM releases the GIL,
which gives near-linear speedup without process-pool serialization overhead.

Methodological references
-------------------------
McLachlan, G., & Peel, D. (2000). Finite mixture models. Wiley.
Nylund, K. L., Asparouhov, T., & Muthen, B. O. (2007). Deciding on the
    number of classes in latent class analysis and growth mixture
    modeling. Structural Equation Modeling, 14(4), 535-569.
Nylund-Gibson, K., & Choi, A. Y. (2018). Ten frequently asked questions
    about latent class analysis. Translational Issues in Psychological
    Science, 4(4), 440-461.
Spurk, D., Hirschi, A., Wang, M., Valero, D., & Kauffeld, S. (2020).
    Latent profile analysis: A review and how-to guide of its application
    within vocational behavior research. Journal of Vocational Behavior,
    120, 103445.
Tikkanen, L., Pyhalto, K., Bujacz, A., & Nieminen, J. (2021). Study
    engagement and burnout of the PhD candidates in medicine. Frontiers
    in Psychology, 12, 727746.

Author: BEFOUM Stephane Richard
Target journal: Learning and Individual Differences (Elsevier)
"""

from __future__ import annotations

import logging
import sys
import warnings
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from configs import CONFIG, ensure_output_directories, set_global_seeds

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# Logging configuration
# ===========================================================================
def configure_logging() -> logging.Logger:
    """Configure structured logging to stdout and a timestamped file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = CONFIG.paths.logs_dir / f"03_lpa_{timestamp}.log"

    logger = logging.getLogger("latent_profile_analysis")
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
# Data loading
# ===========================================================================
def load_analysis_dataset(logger: logging.Logger) -> pd.DataFrame:
    """Load the canonical analysis dataset produced by Script 01.

    If multiple imputation was enabled in Script 01, the first completed
    dataset is used as the reference for profile estimation and a warning
    is logged; full per-imputation profile alignment is out of scope on the
    present (complete) data. With imputation disabled (the default) the
    single canonical dataset is loaded.
    """
    if CONFIG.imputation.enable:
        path = CONFIG.paths.imputed_path(1)
        logger.warning(
            "Imputation is enabled; using imputation 1 as the reference for "
            "profile estimation. Per-imputation alignment is not performed.",
        )
    else:
        path = CONFIG.paths.chinese_phd_dataset

    if not path.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found at {path}. Run "
            f"01_data_preparation.py first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded analysis dataset (N = %d)", len(df))
    return df


def assemble_feature_matrix(
    df: pd.DataFrame, logger: logging.Logger,
) -> tuple[np.ndarray, list[str], pd.Index]:
    """Assemble the standardized profile-feature matrix.

    Returns the feature array, the resolved feature names, and the index of
    the complete-case rows used (so assignments can be mapped back to the
    dataset). The features are already z-scored by Script 01; they are not
    re-standardized here, so the profile means are interpretable directly
    in the original z metric.
    """
    features = list(CONFIG.lpa.profile_features)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Required profile features missing from dataset: {missing}. "
            f"Verify Script 01 produced the standardized *_z columns."
        )

    complete = df[features].dropna()
    n_dropped = len(df) - len(complete)
    if n_dropped > 0:
        logger.warning(
            "%d rows dropped for missing profile features (expected 0 on "
            "the published data)", n_dropped,
        )
    logger.info(
        "Feature matrix: %d cases x %d features (%s)",
        len(complete), len(features), ", ".join(features),
    )
    return complete.to_numpy(dtype=float), features, complete.index


# ===========================================================================
# Single-model fitting and selection criteria
# ===========================================================================
def fit_gmm(
    data: np.ndarray,
    n_components: int,
    n_init: int,
    max_iter: int,
    tol: float,
    random_state: int,
    covariance_type: str = "full",
) -> GaussianMixture:
    """Fit one Gaussian mixture with the configured estimation settings."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        init_params="kmeans",
        reg_covar=1e-6,
    )
    model.fit(data)
    return model


def model_metrics(
    model: GaussianMixture, data: np.ndarray,
) -> dict[str, float]:
    """Compute model-selection criteria for a fitted Gaussian mixture.

    Returns the log-likelihood, parameter count, BIC, AIC, SABIC, and
    normalized entropy in [0, 1] (higher = cleaner classification). SABIC
    uses the sample-size adjustment (N + 2) / 24. BIC and AIC match
    sklearn's built-in methods; they are recomputed here so SABIC and
    entropy share one code path.
    """
    n = data.shape[0]
    ll = float(model.score(data) * n)
    p = int(model._n_parameters())

    bic = -2 * ll + np.log(n) * p
    aic = -2 * ll + 2 * p
    sabic = -2 * ll + np.log((n + 2) / 24) * p

    if model.n_components > 1:
        post = model.predict_proba(data)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_entropy = -np.sum(post * np.log(post + 1e-300))
        entropy = 1 - raw_entropy / (n * np.log(model.n_components))
    else:
        entropy = 1.0

    return {
        "log_likelihood": ll,
        "n_parameters": p,
        "bic": float(bic),
        "aic": float(aic),
        "sabic": float(sabic),
        "entropy": float(entropy),
    }


def fit_model_battery(
    data: np.ndarray, logger: logging.Logger,
) -> tuple[dict[int, GaussianMixture], pd.DataFrame]:
    """Fit Gaussian mixtures across the configured K range.

    Returns the fitted models keyed by K and a selection table with one row
    per K (criteria, minimum profile proportion, convergence flag). Each K
    is fit with the configured number of random starts to avoid local
    maxima (Spurk et al., 2020).
    """
    cfg = CONFIG.lpa
    n = data.shape[0]
    base_seed = (
        CONFIG.reproducibility.root_seed
        + CONFIG.reproducibility.lpa_bootstrap_seed_offset
    )

    models: dict[int, GaussianMixture] = {}
    rows: list[dict[str, Any]] = []

    for k in range(cfg.min_classes, cfg.max_classes + 1):
        try:
            model = fit_gmm(
                data, k,
                n_init=cfg.n_random_starts,
                max_iter=cfg.max_em_iterations,
                tol=cfg.em_convergence_tolerance,
                random_state=base_seed + k,
            )
        except Exception as exc:
            logger.warning("K = %d failed to fit: %s", k, exc)
            continue

        metrics = model_metrics(model, data)
        models[k] = model

        if k > 1:
            labels = model.predict(data)
            proportions = np.bincount(labels, minlength=k) / n
            min_prop = float(proportions.min())
        else:
            min_prop = 1.0

        rows.append({
            "k": k,
            "log_likelihood": round(metrics["log_likelihood"], 2),
            "n_parameters": metrics["n_parameters"],
            "bic": round(metrics["bic"], 2),
            "aic": round(metrics["aic"], 2),
            "sabic": round(metrics["sabic"], 2),
            "entropy": round(metrics["entropy"], 3),
            "min_profile_proportion": round(min_prop, 3),
            "meets_min_size": bool(min_prop >= cfg.min_profile_proportion),
            "converged": bool(model.converged_),
        })
        logger.info(
            "K = %d: BIC = %.1f, SABIC = %.1f, entropy = %.3f, "
            "min profile = %.3f, converged = %s",
            k, metrics["bic"], metrics["sabic"], metrics["entropy"],
            min_prop, model.converged_,
        )

    return models, pd.DataFrame(rows)


# ===========================================================================
# Bootstrap likelihood-ratio test
# ===========================================================================
def bootstrap_lrt(
    data: np.ndarray,
    k_null: int,
    k_alt: int,
    n_bootstrap: int,
    random_state: int,
    logger: logging.Logger,
) -> dict[str, float]:
    """Parametric bootstrap likelihood-ratio test for nested LPA models.

    Tests H0 (k_null classes sufficient) against H1 (k_alt classes fit
    better). Data are repeatedly simulated from the fitted null model;
    both models are refit on each synthetic sample; the bootstrap p-value
    is the proportion of synthetic LR statistics that meet or exceed the
    observed LR (Nylund et al., 2007).
    """
    cfg = CONFIG.lpa
    n = data.shape[0]

    null_model = fit_gmm(
        data, k_null, n_init=cfg.n_random_starts,
        max_iter=cfg.max_em_iterations, tol=cfg.em_convergence_tolerance,
        random_state=random_state,
    )
    alt_model = fit_gmm(
        data, k_alt, n_init=cfg.n_random_starts,
        max_iter=cfg.max_em_iterations, tol=cfg.em_convergence_tolerance,
        random_state=random_state + 1,
    )
    observed_lr = float(2 * (alt_model.score(data) - null_model.score(data)) * n)

    rng = np.random.default_rng(random_state + 2)

    def _iteration(seed: int) -> float:
        try:
            synthetic, _ = null_model.sample(n)
            local = np.random.default_rng(seed)
            local.shuffle(synthetic)
            m_null = fit_gmm(
                synthetic, k_null, n_init=10,
                max_iter=cfg.max_em_iterations,
                tol=cfg.em_convergence_tolerance, random_state=int(seed),
            )
            m_alt = fit_gmm(
                synthetic, k_alt, n_init=10,
                max_iter=cfg.max_em_iterations,
                tol=cfg.em_convergence_tolerance, random_state=int(seed) + 1,
            )
            return float(
                2 * (m_alt.score(synthetic) - m_null.score(synthetic)) * n
            )
        except Exception:
            return float("nan")

    seeds = rng.integers(0, 1_000_000, size=n_bootstrap)
    boot = Parallel(
        n_jobs=CONFIG.hardware.n_cpu_workers,
        backend=CONFIG.hardware.parallel_backend,
    )(delayed(_iteration)(int(s)) for s in seeds)
    valid = [lr for lr in boot if not np.isnan(lr)]
    n_valid = len(valid)

    if n_valid < n_bootstrap * 0.5:
        logger.warning(
            "BLRT %d-vs-%d: only %d / %d replicates converged",
            k_null, k_alt, n_valid, n_bootstrap,
        )

    p_value = (
        float(np.mean([lr >= observed_lr for lr in valid]))
        if n_valid > 0 else float("nan")
    )
    logger.info(
        "BLRT %d-vs-%d: observed LR = %.2f, p = %.4f (n_valid = %d)",
        k_null, k_alt, observed_lr, p_value, n_valid,
    )
    return {
        "k_null": k_null,
        "k_alternative": k_alt,
        "observed_lr": round(observed_lr, 3),
        "p_value": round(p_value, 4) if not np.isnan(p_value) else float("nan"),
        "n_bootstrap_valid": n_valid,
        "favors_k_alternative": bool(not np.isnan(p_value) and p_value < 0.05),
    }


# ===========================================================================
# Optimal-K selection (Spurk hierarchy)
# ===========================================================================
def select_optimal_k(
    selection_df: pd.DataFrame,
    blrt_rows: list[dict[str, float]],
    logger: logging.Logger,
) -> tuple[int, str]:
    """Select the optimal number of profiles using the Spurk hierarchy.

    Decision rule (Spurk et al., 2020; Nylund et al., 2007):
      1. Admissible solutions converge and have all profiles >= 5%.
      2. Among admissible K >= 2, take the BIC minimum as the primary
         candidate.
      3. Cross-check with the BLRT: the largest K for which every step up
         to it is significant gives an upper bound; if the BIC candidate
         exceeds it, fall back to the BLRT bound.
      4. Require entropy >= the configured floor; if the candidate fails,
         step down to the largest admissible K that meets the floor.
      5. Floor of K = 2 (a one-profile "solution" is no profile structure).

    Returns the chosen K and a human-readable justification string.
    """
    cfg = CONFIG.lpa
    admissible = selection_df[
        selection_df["converged"]
        & selection_df["meets_min_size"]
        & (selection_df["k"] >= 2)
    ].copy()

    if len(admissible) == 0:
        logger.warning(
            "No admissible multi-profile solution (all K>=2 had a profile "
            "below %.0f%% or failed to converge); defaulting to K = 2",
            cfg.min_profile_proportion * 100,
        )
        return 2, "fallback: no admissible solution, defaulted to K=2"

    # Step 2: BIC minimum among admissible.
    k_bic = int(admissible.loc[admissible["bic"].idxmin(), "k"])

    # Step 3: BLRT upper bound (largest K with all steps significant).
    blrt_bound = 1
    for k in range(2, int(admissible["k"].max()) + 1):
        step = next((r for r in blrt_rows if r["k_alternative"] == k), None)
        if step is not None and step.get("favors_k_alternative", False):
            blrt_bound = k
        else:
            break
    blrt_bound = max(blrt_bound, 2)

    candidate = min(k_bic, blrt_bound) if blrt_bound >= 2 else k_bic

    # Step 4: entropy floor.
    cand_entropy = float(
        admissible.loc[admissible["k"] == candidate, "entropy"].iloc[0]
    )
    if cand_entropy < cfg.entropy_acceptable:
        meets = admissible[admissible["entropy"] >= cfg.entropy_acceptable]
        if len(meets) > 0:
            stepped = int(meets["k"].max())
            logger.info(
                "Candidate K=%d entropy %.3f below floor %.2f; stepping to "
                "K=%d which meets the entropy floor",
                candidate, cand_entropy, cfg.entropy_acceptable, stepped,
            )
            candidate = stepped

    candidate = max(2, candidate)
    justification = (
        f"BIC minimum K={k_bic}; BLRT upper bound K={blrt_bound}; "
        f"entropy floor {cfg.entropy_acceptable}; selected K={candidate}"
    )
    logger.info("Optimal K selected: %d (%s)", candidate, justification)
    return candidate, justification


# ===========================================================================
# Profile characterization and labeling
# ===========================================================================
def order_profiles_by_wellbeing(
    model: GaussianMixture, feature_names: list[str],
) -> dict[int, int]:
    """Return a relabeling that orders profiles by ascending well-being mean.

    Mixture component indices are arbitrary; reordering them by the
    well-being feature mean makes profile numbering stable and
    interpretable (Profile 1 = lowest well-being). Returns a mapping from
    the original component index to the ordered label (0-based).
    """
    if "wellbeing_z" in feature_names:
        key_idx = feature_names.index("wellbeing_z")
    else:
        key_idx = len(feature_names) - 1
    order = np.argsort(model.means_[:, key_idx])
    return {int(orig): int(new) for new, orig in enumerate(order)}


def characterize_profiles(
    model: GaussianMixture,
    data: np.ndarray,
    feature_names: list[str],
    relabel: dict[int, int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Produce the profile-characterization table and ordered assignments.

    Returns a long-format table (one row per profile x construct, with the
    standardized mean and the profile size/proportion) and the per-case
    profile assignments under the well-being ordering (1-based).
    """
    n = data.shape[0]
    raw_assignments = model.predict(data)
    ordered = np.array([relabel[int(a)] for a in raw_assignments])

    rows: list[dict[str, Any]] = []
    for new_label in range(model.n_components):
        members = ordered == new_label
        size = int(members.sum())
        proportion = size / n
        means = data[members].mean(axis=0) if size > 0 else np.full(
            len(feature_names), np.nan
        )
        for j, feat in enumerate(feature_names):
            rows.append({
                "profile": new_label + 1,
                "construct": feat.replace("_z", ""),
                "standardized_mean": round(float(means[j]), 3),
                "profile_size": size,
                "profile_proportion": round(proportion, 3),
            })
    char = pd.DataFrame(rows)
    return char, ordered + 1


def label_profiles(
    characterization: pd.DataFrame, logger: logging.Logger,
) -> dict[int, str]:
    """Assign provisional descriptive labels from standardized means.

    A construct is called "high" when its profile mean exceeds +0.5 SD and
    "low" below -0.5 SD; profiles with no marked deviation are "average".
    Labels are provisional aids for table presentation; the manuscript's
    final labels should be refined against the qualitative themes (Script
    06) and the integration (Script 07).
    """
    labels: dict[int, str] = {}
    for profile in sorted(characterization["profile"].unique()):
        sub = characterization[characterization["profile"] == profile]
        parts: list[str] = []
        for _, row in sub.iterrows():
            z = row["standardized_mean"]
            if z > 0.5:
                parts.append(f"high {row['construct']}")
            elif z < -0.5:
                parts.append(f"low {row['construct']}")
        labels[int(profile)] = "; ".join(parts) if parts else "average configuration"
        logger.info("Profile %d: %s", profile, labels[int(profile)])
    return labels


# ===========================================================================
# Profile stability via bootstrap
# ===========================================================================
def assess_stability(
    data: np.ndarray, optimal_k: int, logger: logging.Logger,
) -> dict[str, Any]:
    """Assess profile stability by nonparametric bootstrap.

    For each replicate a model is refit on a resampled dataset and its
    predictions on the ORIGINAL data are compared to the reference
    solution's predictions with the Adjusted Rand Coefficient (ARC). High
    mean ARC indicates the profile structure is robust to sampling
    variability.
    """
    cfg = CONFIG.lpa
    n = data.shape[0]
    base_seed = (
        CONFIG.reproducibility.root_seed
        + CONFIG.reproducibility.lpa_bootstrap_seed_offset
        + 9999
    )
    logger.info(
        "Assessing stability with %d bootstrap replicates", cfg.stability_n_bootstrap,
    )

    reference = fit_gmm(
        data, optimal_k, n_init=cfg.n_random_starts,
        max_iter=cfg.max_em_iterations, tol=cfg.em_convergence_tolerance,
        random_state=base_seed,
    )
    reference_labels = reference.predict(data)

    def _iteration(seed: int) -> float:
        local = np.random.default_rng(seed)
        idx = local.integers(0, n, size=n)
        try:
            boot_model = fit_gmm(
                data[idx], optimal_k, n_init=20,
                max_iter=cfg.max_em_iterations,
                tol=cfg.em_convergence_tolerance, random_state=int(seed),
            )
            return float(adjusted_rand_score(
                reference_labels, boot_model.predict(data),
            ))
        except Exception:
            return float("nan")

    seeds = np.random.default_rng(base_seed).integers(
        0, 1_000_000, size=cfg.stability_n_bootstrap,
    )
    arcs = Parallel(
        n_jobs=CONFIG.hardware.n_cpu_workers,
        backend=CONFIG.hardware.parallel_backend,
    )(delayed(_iteration)(int(s)) for s in seeds)
    valid = [a for a in arcs if not np.isnan(a)]
    if len(valid) < cfg.stability_n_bootstrap * 0.5:
        logger.warning(
            "Only %d / %d stability replicates converged",
            len(valid), cfg.stability_n_bootstrap,
        )

    mean_arc = float(np.mean(valid)) if valid else float("nan")
    result = {
        "optimal_k": optimal_k,
        "n_bootstrap_valid": len(valid),
        "mean_arc": round(mean_arc, 3) if not np.isnan(mean_arc) else float("nan"),
        "sd_arc": round(float(np.std(valid, ddof=1)), 3) if len(valid) > 1 else 0.0,
        "median_arc": round(float(np.median(valid)), 3) if valid else float("nan"),
        "arc_ci_lower": round(float(np.percentile(valid, 2.5)), 3) if valid else float("nan"),
        "arc_ci_upper": round(float(np.percentile(valid, 97.5)), 3) if valid else float("nan"),
        "proportion_stable": (
            round(float(np.mean([a >= cfg.stability_min_arc for a in valid])), 3)
            if valid else 0.0
        ),
        "stability_acceptable": bool(
            not np.isnan(mean_arc) and mean_arc >= cfg.stability_min_arc
        ),
    }
    logger.info(
        "Stability: mean ARC = %.3f [%.3f, %.3f], proportion >= %.2f: %.3f",
        result["mean_arc"], result["arc_ci_lower"], result["arc_ci_upper"],
        cfg.stability_min_arc, result["proportion_stable"],
    )
    return result


# ===========================================================================
# Multinomial logistic regression validation
# ===========================================================================
def validate_with_logistic_regression(
    df: pd.DataFrame,
    assignments: np.ndarray,
    case_index: pd.Index,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Validate profile membership with multinomial logistic regression.

    Following Tikkanen et al. (2021), tests whether demographic and
    contextual covariates predict profile membership. statsmodels' MNLogit
    is used when available (it yields per-coefficient standard errors and
    odds ratios); otherwise scikit-learn's multinomial LogisticRegression
    is used and the table reports coefficients and a likelihood-ratio
    model test in place of per-coefficient Wald inference. The engine is
    recorded in the table.
    """
    candidate_covariates = ["Q3a", "Q3b", "Q17", "Q18", "ai_use"]
    covariates = [c for c in candidate_covariates if c in df.columns]
    if not covariates:
        logger.warning("No covariates available for profile validation")
        return pd.DataFrame()

    work = df.loc[case_index, covariates].copy()
    work["profile"] = assignments
    work = work.dropna()
    if work["profile"].nunique() < 2 or len(work) < 50:
        logger.warning("Insufficient data for logistic-regression validation")
        return pd.DataFrame()

    logger.info(
        "Validating profiles via multinomial logistic regression with %d "
        "covariates: %s", len(covariates), covariates,
    )

    X = pd.get_dummies(work[covariates], drop_first=True, dtype=float)
    y = work["profile"].astype(int)

    # Preferred engine: statsmodels MNLogit (Wald inference, odds ratios).
    try:
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import MNLogit

        Xc = sm.add_constant(X, has_constant="add")
        model = MNLogit(y, Xc).fit(disp=False, maxiter=200)
        params, bse, pvals = model.params, model.bse, model.pvalues
        rows: list[dict[str, Any]] = []
        for col in params.columns:
            for cov in params.index:
                coef = float(params.loc[cov, col])
                se = float(bse.loc[cov, col])
                p = float(pvals.loc[cov, col])
                rows.append({
                    "engine": "statsmodels_mnlogit",
                    "profile_vs_base": str(col),
                    "covariate": str(cov),
                    "coef": round(coef, 4),
                    "se": round(se, 4),
                    "odds_ratio": round(float(np.exp(coef)), 3),
                    "or_ci_lower": round(float(np.exp(coef - 1.96 * se)), 3),
                    "or_ci_upper": round(float(np.exp(coef + 1.96 * se)), 3),
                    "p_value": round(p, 4),
                    "significant": bool(p < 0.05),
                })
        logger.info("Profile validation fit via statsmodels MNLogit")
        return pd.DataFrame(rows)
    except ImportError:
        pass  # fall through to sklearn

    # Fallback engine: scikit-learn multinomial logistic regression.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.to_numpy())
    # Modern scikit-learn (>= 1.7) handles multinomial logistic regression
    # natively for multi-class targets; the former ``multi_class`` argument
    # was removed. A large C makes the fit near-unpenalized so coefficients
    # are interpretable in the inference sense.
    clf = LogisticRegression(
        solver="lbfgs", max_iter=2000, C=1e6,
        random_state=CONFIG.reproducibility.root_seed,
    )
    clf.fit(X_scaled, y)

    # Likelihood-ratio test against an intercept-only model.
    from sklearn.metrics import log_loss
    full_ll = -log_loss(y, clf.predict_proba(X_scaled), normalize=False)
    base_probs = np.tile(
        np.bincount(
            pd.factorize(y)[0], minlength=y.nunique()
        ) / len(y),
        (len(y), 1),
    )
    null_ll = -log_loss(y, base_probs, normalize=False)
    lr_stat = float(2 * (full_ll - null_ll))
    df_lr = int(X.shape[1] * (y.nunique() - 1))
    from scipy import stats as scipy_stats
    lr_p = float(scipy_stats.chi2.sf(lr_stat, df_lr)) if df_lr > 0 else float("nan")

    classes = list(clf.classes_)
    feature_names = list(X.columns)
    rows = []
    # sklearn coefficients are on standardized predictors; rescale to raw.
    scales = scaler.scale_
    for ci, cls in enumerate(classes):
        if len(classes) == 2 and ci == 0:
            continue  # binary case: sklearn stores one coefficient row
        coef_row = clf.coef_[ci] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        for fi, fname in enumerate(feature_names):
            coef_raw = float(coef_row[fi] / scales[fi]) if scales[fi] else float("nan")
            rows.append({
                "engine": "sklearn_logreg",
                "profile_vs_base": f"profile_{cls}",
                "covariate": fname,
                "coef": round(coef_raw, 4),
                "se": np.nan,
                "odds_ratio": round(float(np.exp(coef_raw)), 3),
                "or_ci_lower": np.nan,
                "or_ci_upper": np.nan,
                "p_value": np.nan,
                "significant": np.nan,
            })
    out = pd.DataFrame(rows)
    out.attrs["lr_stat"] = lr_stat
    out.attrs["lr_p"] = lr_p
    logger.info(
        "Profile validation fit via sklearn multinomial logistic regression; "
        "model LR chi2(%d) = %.2f, p = %.4f (per-coefficient Wald inference "
        "unavailable in fallback)",
        df_lr, lr_stat, lr_p,
    )
    return out


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
    """Execute the full latent-profile-analysis pipeline."""
    ensure_output_directories()
    set_global_seeds()
    logger = configure_logging()

    logger.info("Pipeline version: %s", CONFIG.pipeline_version)
    logger.info("Root seed: %d", CONFIG.reproducibility.root_seed)
    logger.info("Random starts per model: %d", CONFIG.lpa.n_random_starts)
    logger.info("BLRT bootstrap iterations: %d", CONFIG.lpa.blrt_n_bootstrap)
    logger.info("Stage: 03_latent_profile_analysis")

    try:
        df = load_analysis_dataset(logger)
        data, feature_names, case_index = assemble_feature_matrix(df, logger)

        # --- Phase 1: Fit the K battery ---
        logger.info("=" * 72)
        logger.info("PHASE 1: Model battery (K = %d..%d)",
                    CONFIG.lpa.min_classes, CONFIG.lpa.max_classes)
        logger.info("=" * 72)
        models, selection_df = fit_model_battery(data, logger)
        write_table(selection_df, "table_s9_lpa_selection.csv", logger)

        # --- Phase 2: BLRT ---
        logger.info("=" * 72)
        logger.info("PHASE 2: Bootstrap likelihood-ratio test")
        logger.info("=" * 72)
        blrt_rows: list[dict[str, float]] = []
        if CONFIG.lpa.use_blrt:
            for k in range(2, CONFIG.lpa.max_classes + 1):
                if k in models and (k - 1) in models:
                    blrt_rows.append(bootstrap_lrt(
                        data, k - 1, k,
                        n_bootstrap=CONFIG.lpa.blrt_n_bootstrap,
                        random_state=CONFIG.reproducibility.root_seed + k,
                        logger=logger,
                    ))
        if blrt_rows:
            write_table(pd.DataFrame(blrt_rows), "table_s10_lpa_blrt.csv", logger)

        # --- Phase 3: Select optimal K ---
        logger.info("=" * 72)
        logger.info("PHASE 3: Optimal-K selection (Spurk hierarchy)")
        logger.info("=" * 72)
        optimal_k, justification = select_optimal_k(
            selection_df, blrt_rows, logger,
        )
        if optimal_k not in models:
            logger.warning(
                "Optimal K=%d has no fitted model; refitting", optimal_k,
            )
            models[optimal_k] = fit_gmm(
                data, optimal_k, n_init=CONFIG.lpa.n_random_starts,
                max_iter=CONFIG.lpa.max_em_iterations,
                tol=CONFIG.lpa.em_convergence_tolerance,
                random_state=CONFIG.reproducibility.root_seed,
            )

        # --- Phase 4: Characterize the chosen solution ---
        logger.info("=" * 72)
        logger.info("PHASE 4: Profile characterization (K = %d)", optimal_k)
        logger.info("=" * 72)
        chosen_model = models[optimal_k]
        relabel = order_profiles_by_wellbeing(chosen_model, feature_names)
        characterization, assignments = characterize_profiles(
            chosen_model, data, feature_names, relabel, logger,
        )
        write_table(
            characterization, "table_s11_profile_characterization.csv", logger,
        )
        profile_labels = label_profiles(characterization, logger)

        # --- Phase 5: Stability ---
        logger.info("=" * 72)
        logger.info("PHASE 5: Profile stability")
        logger.info("=" * 72)
        stability = assess_stability(data, optimal_k, logger)
        write_table(pd.DataFrame([stability]), "table_s12_profile_stability.csv", logger)

        # --- Phase 6: Logistic-regression validation ---
        logger.info("=" * 72)
        logger.info("PHASE 6: Multinomial logistic-regression validation")
        logger.info("=" * 72)
        validation = pd.DataFrame()
        if CONFIG.lpa.validate_with_logistic_regression:
            validation = validate_with_logistic_regression(
                df, assignments, case_index, logger,
            )
            if len(validation) > 0:
                write_table(
                    validation, "table_s13_profile_validation_logistic.csv", logger,
                )

        # --- Phase 7: Persist assignments and posteriors for downstream use ---
        logger.info("=" * 72)
        logger.info("PHASE 7: Persisting profile assignments")
        logger.info("=" * 72)
        posteriors = chosen_model.predict_proba(data)
        # Reorder posterior columns to match the well-being ordering.
        ordered_post = np.empty_like(posteriors)
        for orig, new in relabel.items():
            ordered_post[:, new] = posteriors[:, orig]
        max_post = ordered_post.max(axis=1)

        assignment_df = pd.DataFrame({
            "case_index": case_index.to_numpy(),
            "profile": assignments,
            "max_posterior": np.round(max_post, 3),
            "profile_label": [profile_labels[int(p)] for p in assignments],
        })
        # Attach the source respondent id if present, for traceability.
        if "respid" in df.columns:
            assignment_df["respid"] = df.loc[case_index, "respid"].to_numpy()

        assignment_path = (
            CONFIG.paths.models_dir / "profile_assignments.csv"
        )
        assignment_df.to_csv(assignment_path, index=False)
        logger.info(
            "Profile assignments written to %s (%d rows)",
            assignment_path, len(assignment_df),
        )

        # Selection metadata for the manuscript and downstream scripts.
        meta = {
            "optimal_k": optimal_k,
            "selection_justification": justification,
            "mean_classification_posterior": round(float(max_post.mean()), 3),
            "entropy": float(
                selection_df.loc[selection_df["k"] == optimal_k, "entropy"].iloc[0]
            ),
            "mean_arc": stability["mean_arc"],
            "stability_acceptable": stability["stability_acceptable"],
            "profile_labels": profile_labels,
        }
        meta_path = CONFIG.paths.models_dir / "profile_solution_meta.json"
        import json
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        logger.info("Profile solution metadata written to %s", meta_path)

        # --- Final summary ---
        logger.info("=" * 72)
        logger.info("Latent profile analysis completed")
        logger.info("=" * 72)
        sizes = (
            characterization.drop_duplicates("profile")
            .set_index("profile")["profile_size"].to_dict()
        )
        logger.info(
            "Optimal K = %d | profile sizes = %s | mean ARC = %.3f | "
            "mean posterior = %.3f",
            optimal_k, sizes, stability["mean_arc"],
            float(max_post.mean()),
        )
        for p, lbl in profile_labels.items():
            logger.info("  Profile %d (n=%d): %s", p, sizes.get(p, 0), lbl)
        return 0

    except Exception as exc:
        logger.exception("Latent profile analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
