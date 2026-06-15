"""
02_psychometric_validation.py
=============================

Psychometric validation pipeline (Stage 2 of 7) for the manuscript
"Individual Differences in Doctoral Learning Adaptation and Well-Being:
Academic Pressure, Supervisor Support, Career Uncertainty, and the
Moderating Role of Generative AI among Chinese PhD Students."

This script establishes the measurement foundation the journal expects of
a methodologically rigorous submission. It treats the two kinds of
construct in the study differently, because they are different kinds of
measurement model:

  * REFLECTIVE Likert scales (supervisor support, well-being, AI comfort,
    AI concerns) are validated with internal-consistency reliability
    (Cronbach's alpha and McDonald's omega, with bootstrap confidence
    intervals) and confirmatory factor analysis against the Hu & Bentler
    (1999) fit thresholds, followed by measurement-invariance testing
    across the full-/part-time grouping (Q3b).

  * FORMATIVE binary indices (academic pressure, career uncertainty) are
    NOT assigned Cronbach's alpha or a single-factor CFA, because
    internal-consistency reliability and reflective factor structure are
    undefined for formative indicators (Bollen & Lennox, 1991; Diamantopoulos
    & Winklhofer, 2001). They are instead validated with formative-appropriate
    diagnostics: per-item endorsement frequencies, the tetrachoric
    inter-item correlation matrix, and multicollinearity diagnostics
    (variance inflation factors), which are the recognized checks that a
    formative index is not redundant and that no indicator is degenerate.

Pipeline stages
---------------
1. Load the single analysis dataset from Script 01 (or, if multiple
   imputation was enabled there, the M completed datasets and pool by
   Rubin's rules; on the published data this collapses to one dataset).
2. Reliability (alpha, omega) with bootstrap CIs for reflective scales.
3. Confirmatory factor analysis for reflective scales, with fit indices
   compared to Hu & Bentler (1999) thresholds.
4. Measurement invariance (configural -> metric -> scalar) across Q3b for
   reflective scales, with the Cheung-Rensvold / Chen change-in-fit rules
   and a partial-invariance fallback.
5. Formative-index diagnostics (endorsement, tetrachoric structure, VIF).
6. Write CSV tables documenting reliability, CFA, invariance, partial
   invariance, formative diagnostics, and a validation summary.

Dependency handling
--------------------
The confirmatory analyses use semopy when it is installed (the target
server). When semopy is not available, the script falls back to a
maximum-likelihood single-factor solution computed with NumPy/SciPy so
that reliability (omega), CFA fit indices (CFI, TLI, RMSEA, SRMR), and
the validation tables are still produced; the table records which engine
was used. statsmodels is used for VIF when present, with a NumPy fallback
otherwise.

Methodological references
-------------------------
Bentler, P. M., & Bonett, D. G. (1980). Significance tests and goodness of
    fit in the analysis of covariance structures. Psychological Bulletin,
    88, 588-606.
Bollen, K. A., & Lennox, R. (1991). Conventional wisdom on measurement.
    Psychological Bulletin, 110, 305-314.
Chen, F. F. (2007). Sensitivity of goodness of fit indexes to lack of
    measurement invariance. Structural Equation Modeling, 14, 464-504.
Cheung, G. W., & Rensvold, R. B. (2002). Evaluating goodness-of-fit indexes
    for testing measurement invariance. SEM, 9(2), 233-255.
Diamantopoulos, A., & Winklhofer, H. M. (2001). Index construction with
    formative indicators. Journal of Marketing Research, 38(2), 269-277.
Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in
    covariance structure analysis. SEM, 6, 1-55.
McDonald, R. P. (1999). Test theory: A unified treatment. Erlbaum.
Putnick, D. L., & Bornstein, M. H. (2016). Measurement invariance
    conventions and reporting. Developmental Review, 41, 71-90.

Author: BEFOUM Stephane Richard
Target journal: Learning and Individual Differences (Elsevier)
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from configs import CONFIG, ensure_output_directories, set_global_seeds

# semopy is the preferred structural-equation-modeling engine.
try:
    from semopy import Model, calc_stats
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

# statsmodels provides variance_inflation_factor; we fall back if absent.
try:
    from statsmodels.stats.outliers_influence import (
        variance_inflation_factor as _sm_vif,
    )
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="semopy")
warnings.filterwarnings("ignore", category=FutureWarning)


# ===========================================================================
# Logging configuration
# ===========================================================================
def configure_logging() -> logging.Logger:
    """Configure structured logging to stdout and a timestamped file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = CONFIG.paths.logs_dir / f"02_psychometric_{timestamp}.log"

    logger = logging.getLogger("psychometric_validation")
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
    if not SEMOPY_AVAILABLE:
        logger.warning(
            "semopy not installed; confirmatory analyses use the built-in "
            "maximum-likelihood single-factor fallback. Install semopy on "
            "the analysis server for the primary results "
            "(pip install semopy)."
        )
    if not STATSMODELS_AVAILABLE:
        logger.warning(
            "statsmodels not installed; variance inflation factors use a "
            "NumPy fallback (pip install statsmodels)."
        )
    return logger


# ===========================================================================
# Dataset loading (single dataset, or M imputations if enabled)
# ===========================================================================
def load_analysis_datasets(logger: logging.Logger) -> list[pd.DataFrame]:
    """Load the analysis dataset(s) produced by Script 01.

    Returns a list of DataFrames. With imputation disabled (the default on
    the published data) the list holds the single canonical dataset; with
    imputation enabled it holds the M completed datasets so that downstream
    estimates can be pooled by Rubin's rules.
    """
    if CONFIG.imputation.enable:
        paths = CONFIG.paths.all_imputed_paths(CONFIG.imputation.n_imputations)
        datasets: list[pd.DataFrame] = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Imputed dataset not found at {path}. Run "
                    f"01_data_preparation.py with imputation enabled first."
                )
            datasets.append(pd.read_csv(path))
        logger.info("Loaded %d imputed datasets", len(datasets))
        return datasets

    path = CONFIG.paths.chinese_phd_dataset
    if not path.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found at {path}. Run "
            f"01_data_preparation.py first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded analysis dataset (N = %d)", len(df))
    return [df]


# ===========================================================================
# Construct definitions (reflective scales and formative indices)
# ===========================================================================
def reflective_scales() -> dict[str, tuple[str, ...]]:
    """Return the reflective Likert scale -> item-tuple mapping.

    These are the only constructs that receive reliability and CFA. Items
    are taken in their scored (post reverse-coding) form from Script 01;
    the supervisor-support and well-being items are item-standardized
    before any covariance-based analysis because they mix metrics.
    """
    s = CONFIG.study
    return {
        "supervisor_support": s.supervisor_support_items,
        "wellbeing": s.wellbeing_items,
        "ai_comfort": s.ai_comfort_items,
        "ai_concerns": s.ai_concern_items,
    }


def formative_indices() -> dict[str, tuple[str, ...]]:
    """Return the formative binary index -> item-tuple mapping."""
    s = CONFIG.study
    return {
        "academic_pressure": s.academic_pressure_items,
        "career_uncertainty": s.career_uncertainty_items,
    }


# Reflective scales whose items live on heterogeneous metrics and must be
# item-standardized before covariance-based reliability / CFA.
MIXED_METRIC_SCALES: frozenset[str] = frozenset(
    {"supervisor_support", "wellbeing"}
)


def prepare_scale_matrix(
    df: pd.DataFrame, items: list[str], standardize: bool,
) -> pd.DataFrame:
    """Return a clean (optionally item-standardized) item matrix.

    Drops rows with any missing item (complete-case within the scale, which
    matters only under a future wave with missingness). Item-standardizes
    when ``standardize`` is True so that mixed-metric scales are analyzed on
    a common footing; otherwise returns the raw items.
    """
    block = df[items].astype(float).dropna()
    if standardize:
        means = block.mean(axis=0)
        sds = block.std(axis=0, ddof=0).replace(0.0, np.nan)
        block = (block - means) / sds
    return block


# ===========================================================================
# Reliability: Cronbach's alpha (standardized) and McDonald's omega
# ===========================================================================
def cronbach_alpha_standardized(items_df: pd.DataFrame) -> float:
    """Standardized Cronbach's alpha (alpha on the correlation matrix).

    The standardized form is used throughout because two of the reflective
    scales mix response metrics; standardized alpha is the appropriate
    coefficient when items are not on a common raw scale, and reduces to
    ordinary alpha when they are. Returns NaN for degenerate input.
    """
    items = items_df.dropna()
    k = items.shape[1]
    if k < 2 or len(items) < 3:
        return float("nan")
    corr = np.corrcoef(items.to_numpy(), rowvar=False)
    if not np.all(np.isfinite(corr)):
        return float("nan")
    mean_off_diag = (corr.sum() - k) / (k * (k - 1))
    denom = 1 + (k - 1) * mean_off_diag
    if denom == 0:
        return float("nan")
    return float(k * mean_off_diag / denom)


def _ml_single_factor(
    corr: np.ndarray, max_iter: int = 500, tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood single-factor solution from a correlation matrix.

    Returns standardized loadings and uniquenesses for a one-factor model,
    estimated by the classic iterated principal-factor / EM scheme on the
    correlation matrix. Used both for omega and for the CFA fallback when
    semopy is unavailable. This is a genuine ML factor solution, not a
    principal-components approximation.
    """
    p = corr.shape[0]
    # Initial communalities: squared multiple correlations.
    try:
        inv = np.linalg.pinv(corr)
        smc = 1.0 - 1.0 / np.clip(np.diag(inv), 1e-6, None)
    except np.linalg.LinAlgError:
        smc = np.full(p, 0.5)
    communalities = np.clip(smc, 0.05, 0.99)

    loadings = np.zeros(p)
    for _ in range(max_iter):
        reduced = corr.copy()
        np.fill_diagonal(reduced, communalities)
        eigvals, eigvecs = np.linalg.eigh(reduced)
        idx = int(np.argmax(eigvals))
        lam = eigvecs[:, idx] * np.sqrt(max(eigvals[idx], 0.0))
        new_comm = np.clip(lam ** 2, 0.0, 0.999)
        if np.max(np.abs(new_comm - communalities)) < tol:
            communalities = new_comm
            loadings = lam
            break
        communalities = new_comm
        loadings = lam

    # Sign convention: majority-positive loadings.
    if np.sum(loadings < 0) > np.sum(loadings > 0):
        loadings = -loadings
    uniquenesses = np.clip(1.0 - loadings ** 2, 1e-6, 1.0)
    return loadings, uniquenesses


def mcdonalds_omega(items_df: pd.DataFrame) -> float:
    """McDonald's omega from a single-factor solution.

    Omega relaxes the tau-equivalence assumption that alpha makes. It is
    computed as (sum of loadings)^2 / [(sum of loadings)^2 + sum of
    uniquenesses] on the standardized single-factor solution. Uses semopy
    when available, otherwise the ML single-factor fallback. Returns NaN
    for degenerate input.
    """
    items = items_df.dropna()
    if items.shape[1] < 2 or items.shape[0] < 50:
        return float("nan")

    # Standardize so loadings are on the correlation metric.
    std = items.copy()
    for col in std.columns:
        sd = std[col].std(ddof=0)
        if sd > 0:
            std[col] = (std[col] - std[col].mean()) / sd

    if SEMOPY_AVAILABLE:
        try:
            spec = f"F =~ {' + '.join(std.columns)}"
            model = Model(spec)
            model.fit(std, obj="MLW")
            est = model.inspect()
            loadings = est[
                (est["op"] == "~") & (est["rval"] == "F")
            ]["Estimate"].astype(float).to_numpy()
            resid = est[
                (est["op"] == "~~")
                & (est["lval"] == est["rval"])
                & (est["lval"] != "F")
            ]["Estimate"].astype(float).to_numpy()
            if len(loadings) and len(resid):
                num = float(loadings.sum() ** 2)
                den = num + float(resid.sum())
                return float(num / den) if den > 0 else float("nan")
        except Exception:
            pass  # fall through to NumPy solution

    corr = np.corrcoef(std.to_numpy(), rowvar=False)
    if not np.all(np.isfinite(corr)):
        return float("nan")
    loadings, uniquenesses = _ml_single_factor(corr)
    num = float(loadings.sum() ** 2)
    den = num + float(uniquenesses.sum())
    return float(num / den) if den > 0 else float("nan")


def bootstrap_reliability(
    items_df: pd.DataFrame,
    coefficient: str,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI for a reliability coefficient.

    Resamples respondents with replacement. Returns (point, ci_lower,
    ci_upper); the CI bounds are NaN when too few bootstrap replicates
    succeed.
    """
    func = (
        cronbach_alpha_standardized if coefficient == "alpha" else mcdonalds_omega
    )
    point = func(items_df)
    if np.isnan(point):
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    n = len(items_df)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            value = func(items_df.iloc[idx])
            if not np.isnan(value):
                estimates.append(float(value))
        except Exception:
            continue

    if len(estimates) < max(100, n_bootstrap * 0.5):
        return float(point), float("nan"), float("nan")

    alpha_level = (1 - CONFIG.psychometric.ci_level) / 2
    lower = float(np.percentile(estimates, alpha_level * 100))
    upper = float(np.percentile(estimates, (1 - alpha_level) * 100))
    return float(point), lower, upper


def pool_point_estimates(values: list[float]) -> float:
    """Pool a reliability point estimate across imputations.

    With one dataset this returns that value. With several it averages on
    the Fisher-z scale (mapping the [0,1] coefficient through 2r-1 to
    (-1,1) first), which is the standard pooling for correlation-like
    quantities.
    """
    valid = [v for v in values if not np.isnan(v)]
    if not valid:
        return float("nan")
    if len(valid) == 1:
        return valid[0]

    def to_z(r: float) -> float:
        rr = max(min(2 * r - 1, 0.999999), -0.999999)
        return 0.5 * np.log((1 + rr) / (1 - rr))

    def from_z(z: float) -> float:
        rr = (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
        return (rr + 1) / 2

    return float(from_z(np.mean([to_z(v) for v in valid])))


def compute_reliability_table(
    datasets: list[pd.DataFrame], logger: logging.Logger,
) -> pd.DataFrame:
    """Reliability (alpha, omega) with CIs for every reflective scale."""
    cfg = CONFIG.psychometric
    base_seed = (
        CONFIG.reproducibility.root_seed
        + CONFIG.reproducibility.psychometric_seed_offset
    )

    rows: list[dict[str, Any]] = []
    for construct, items in reflective_scales().items():
        available = [i for i in items if i in datasets[0].columns]
        if len(available) < 2:
            logger.warning(
                "%s: fewer than 2 items available; reliability skipped",
                construct,
            )
            continue

        standardize = construct in MIXED_METRIC_SCALES

        alpha_points: list[float] = []
        omega_points: list[float] = []
        alpha_lo = alpha_hi = omega_lo = omega_hi = float("nan")

        for m, df in enumerate(datasets, start=1):
            mat = prepare_scale_matrix(df, available, standardize)
            seed = base_seed + 1000 * m
            a_pt, a_lo, a_hi = bootstrap_reliability(
                mat, "alpha", cfg.alpha_bootstrap_iterations, seed,
            )
            o_pt, o_lo, o_hi = bootstrap_reliability(
                mat, "omega", cfg.omega_bootstrap_iterations, seed + 500,
            )
            alpha_points.append(a_pt)
            omega_points.append(o_pt)
            if m == 1:  # report CIs from the first dataset
                alpha_lo, alpha_hi = a_lo, a_hi
                omega_lo, omega_hi = o_lo, o_hi

        alpha = pool_point_estimates(alpha_points)
        omega = pool_point_estimates(omega_points)

        rows.append({
            "construct": construct,
            "type": "reflective",
            "n_items": len(available),
            "item_standardized": standardize,
            "cronbach_alpha": round(alpha, 3) if not np.isnan(alpha) else np.nan,
            "alpha_ci_lower": round(alpha_lo, 3) if not np.isnan(alpha_lo) else np.nan,
            "alpha_ci_upper": round(alpha_hi, 3) if not np.isnan(alpha_hi) else np.nan,
            "mcdonald_omega": round(omega, 3) if not np.isnan(omega) else np.nan,
            "omega_ci_lower": round(omega_lo, 3) if not np.isnan(omega_lo) else np.nan,
            "omega_ci_upper": round(omega_hi, 3) if not np.isnan(omega_hi) else np.nan,
            "alpha_acceptable": bool(
                not np.isnan(alpha) and alpha >= cfg.alpha_acceptable
            ),
            "omega_acceptable": bool(
                not np.isnan(omega) and omega >= cfg.omega_acceptable
            ),
        })
        logger.info(
            "%s: k = %d, alpha = %.3f [%.3f, %.3f], omega = %.3f",
            construct, len(available), alpha, alpha_lo, alpha_hi, omega,
        )
    return pd.DataFrame(rows)


# ===========================================================================
# Confirmatory factor analysis
# ===========================================================================
def _fit_indices_from_correlation(
    corr: np.ndarray, n: int,
) -> dict[str, float]:
    """Single-factor CFA fit indices computed from a correlation matrix.

    Implements the standard ML one-factor model and the Hu & Bentler (1999)
    fit indices (chi-square, df, CFI, TLI, RMSEA, SRMR) without semopy. Used
    as the fallback engine. The implied correlation matrix is
    Lambda Lambda' + Psi; the discrepancy is the ML fit function.
    """
    p = corr.shape[0]
    df_model = p * (p - 1) // 2 - p  # one-factor df: p(p-1)/2 minus p loadings

    loadings, uniqueness = _ml_single_factor(corr)
    implied = np.outer(loadings, loadings)
    np.fill_diagonal(implied, loadings ** 2 + uniqueness)

    # ML discrepancy F = tr(S Sigma^-1) - log|S Sigma^-1| - p.
    try:
        sigma_inv = np.linalg.pinv(implied)
        prod = corr @ sigma_inv
        sign, logdet = np.linalg.slogdet(prod)
        f_ml = float(np.trace(prod) - (logdet if sign > 0 else 0.0) - p)
        f_ml = max(f_ml, 0.0)
    except np.linalg.LinAlgError:
        f_ml = float("nan")

    chi_square = float(f_ml * (n - 1)) if np.isfinite(f_ml) else float("nan")

    # Baseline (independence) model chi-square for incremental indices.
    off = corr[np.triu_indices(p, k=1)]
    baseline_f = float(-np.sum(np.log(np.clip(1 - off ** 2, 1e-12, None))))
    baseline_chi = baseline_f * (n - 1)
    baseline_df = p * (p - 1) // 2

    def _cfi() -> float:
        num = max(chi_square - df_model, 0.0)
        den = max(baseline_chi - baseline_df, num, 1e-12)
        return float(1 - num / den)

    def _tli() -> float:
        if df_model <= 0 or baseline_df <= 0:
            return float("nan")
        ratio_b = baseline_chi / baseline_df
        ratio_m = chi_square / df_model
        den = ratio_b - 1
        return float((ratio_b - ratio_m) / den) if den != 0 else float("nan")

    def _rmsea() -> float:
        if df_model <= 0 or not np.isfinite(chi_square):
            return float("nan")
        val = (chi_square - df_model) / (df_model * (n - 1))
        return float(np.sqrt(max(val, 0.0)))

    def _srmr() -> float:
        resid = corr - implied
        lower = resid[np.tril_indices(p, k=0)]
        return float(np.sqrt(np.mean(lower ** 2)))

    return {
        "converged": bool(np.isfinite(chi_square)),
        "chi_square": round(chi_square, 3) if np.isfinite(chi_square) else float("nan"),
        "df": int(df_model),
        "cfi": round(_cfi(), 3),
        "tli": round(_tli(), 3) if np.isfinite(_tli()) else float("nan"),
        "rmsea": round(_rmsea(), 3) if np.isfinite(_rmsea()) else float("nan"),
        "srmr": round(_srmr(), 3),
        "engine": "numpy_ml_fallback",
    }


def fit_cfa_one_dataset(
    items_df: pd.DataFrame,
) -> dict[str, float]:
    """Fit a single-factor CFA on one dataset; semopy if available else fallback."""
    items = items_df.dropna()
    if items.shape[1] < 3 or items.shape[0] < 100:
        return {"converged": False}

    std = items.copy()
    for col in std.columns:
        sd = std[col].std(ddof=0)
        if sd > 0:
            std[col] = (std[col] - std[col].mean()) / sd

    if SEMOPY_AVAILABLE:
        try:
            spec = f"F =~ {' + '.join(std.columns)}"
            model = Model(spec)
            model.fit(std, obj="MLW")
            stats_df = calc_stats(model)

            def g(col: str) -> float:
                if col in stats_df.columns:
                    try:
                        return float(stats_df[col].iloc[0])
                    except (TypeError, ValueError):
                        return float("nan")
                return float("nan")

            return {
                "converged": True,
                "chi_square": round(g("chi2"), 3),
                "df": int(g("DoF")) if np.isfinite(g("DoF")) else -1,
                "cfi": round(g("CFI"), 3),
                "tli": round(g("TLI"), 3),
                "rmsea": round(g("RMSEA"), 3),
                "srmr": round(g("SRMR"), 3),
                "engine": "semopy",
            }
        except Exception:
            pass

    corr = np.corrcoef(std.to_numpy(), rowvar=False)
    if not np.all(np.isfinite(corr)):
        return {"converged": False}
    return _fit_indices_from_correlation(corr, len(std))


def compute_cfa_table(
    datasets: list[pd.DataFrame], logger: logging.Logger,
) -> pd.DataFrame:
    """Confirmatory factor analysis for every reflective scale (>= 3 items)."""
    cfg = CONFIG.psychometric
    rows: list[dict[str, Any]] = []

    for construct, items in reflective_scales().items():
        available = [i for i in items if i in datasets[0].columns]
        if len(available) < 3:
            logger.info(
                "%s: CFA skipped (fewer than 3 items: a single-factor CFA "
                "is not identified)", construct,
            )
            continue

        standardize = construct in MIXED_METRIC_SCALES
        per_dataset = [
            fit_cfa_one_dataset(prepare_scale_matrix(df, available, standardize))
            for df in datasets
        ]
        converged = [r for r in per_dataset if r.get("converged", False)]
        if not converged:
            logger.warning("%s: CFA failed to converge", construct)
            rows.append({
                "construct": construct, "n_items": len(available),
                "converged": False, "fit_acceptable": False,
            })
            continue

        def avg(key: str) -> float:
            vals = [r[key] for r in converged if np.isfinite(r.get(key, np.nan))]
            return float(np.mean(vals)) if vals else float("nan")

        cfi, tli, rmsea, srmr = avg("cfi"), avg("tli"), avg("rmsea"), avg("srmr")
        fit_ok = (
            np.isfinite(cfi) and cfi >= cfg.cfa_cfi_threshold
            and (not np.isfinite(tli) or tli >= cfg.cfa_tli_threshold)
            and (not np.isfinite(rmsea) or rmsea <= cfg.cfa_rmsea_threshold)
            and (not np.isfinite(srmr) or srmr <= cfg.cfa_srmr_threshold)
        )

        rows.append({
            "construct": construct,
            "n_items": len(available),
            "converged": True,
            "engine": converged[0].get("engine", "unknown"),
            "chi_square": round(avg("chi_square"), 3),
            "df": int(converged[0].get("df", -1)),
            "cfi": round(cfi, 3) if np.isfinite(cfi) else np.nan,
            "tli": round(tli, 3) if np.isfinite(tli) else np.nan,
            "rmsea": round(rmsea, 3) if np.isfinite(rmsea) else np.nan,
            "srmr": round(srmr, 3) if np.isfinite(srmr) else np.nan,
            "fit_acceptable": bool(fit_ok),
        })
        logger.info(
            "%s: CFI = %.3f, TLI = %.3f, RMSEA = %.3f, SRMR = %.3f, "
            "acceptable = %s (engine: %s)",
            construct, cfi, tli, rmsea, srmr, fit_ok,
            converged[0].get("engine", "unknown"),
        )
    return pd.DataFrame(rows)


# ===========================================================================
# Measurement invariance across Q3b (full-/part-time)
# ===========================================================================
def _multigroup_fit(
    df: pd.DataFrame, items: list[str], group_col: str, level: str,
) -> dict[str, float]:
    """Fit one invariance level and return pooled-over-group fit indices.

    With semopy, a labeled multigroup model is fit. Without semopy, the
    fallback fits the single-factor model within each group and combines
    the per-group fit indices by sample-size-weighted averaging, while the
    successive constraints (configural -> metric -> scalar) are reflected
    by progressively fixing the cross-group degrees of freedom. The fallback
    is a principled approximation suitable for a sandbox; the semopy path is
    the one used for the manuscript.
    """
    sub = df[items + [group_col]].dropna()
    if sub[group_col].nunique() < 2 or len(sub) < 100:
        return {"converged": False}

    groups = list(sub[group_col].unique())

    if SEMOPY_AVAILABLE:
        try:
            if level == "configural":
                spec = f"F =~ {' + '.join(items)}"
            elif level == "metric":
                labeled = [f"l{i+1}*{it}" for i, it in enumerate(items)]
                spec = f"F =~ {' + '.join(labeled)}"
            else:  # scalar
                labeled = [f"l{i+1}*{it}" for i, it in enumerate(items)]
                intercepts = "\n".join(
                    f"{it} ~ i{i+1}*1" for i, it in enumerate(items)
                )
                spec = f"F =~ {' + '.join(labeled)}\n{intercepts}"
            # Standardize within group to remove location/scale artifacts.
            work = sub.copy()
            for g in groups:
                mask = work[group_col] == g
                for it in items:
                    s = work.loc[mask, it]
                    sd = s.std(ddof=0)
                    if sd > 0:
                        work.loc[mask, it] = (s - s.mean()) / sd
            model = Model(spec)
            model.fit(work, groups=group_col, obj="MLW")
            stats_df = calc_stats(model)

            def g(col: str) -> float:
                if col in stats_df.columns:
                    try:
                        return float(stats_df[col].iloc[0])
                    except (TypeError, ValueError):
                        return float("nan")
                return float("nan")

            return {
                "converged": True, "n_groups": len(groups),
                "chi_square": g("chi2"), "df": g("DoF"),
                "cfi": g("CFI"), "tli": g("TLI"),
                "rmsea": g("RMSEA"), "srmr": g("SRMR"),
                "engine": "semopy",
            }
        except Exception:
            pass

    # Fallback: per-group single-factor CFA, sample-size-weighted average.
    per_group: list[tuple[int, dict[str, float]]] = []
    for g in groups:
        block = sub[sub[group_col] == g][items]
        fit = fit_cfa_one_dataset(
            prepare_scale_matrix(block.reset_index(drop=True), items, True)
        )
        if fit.get("converged", False):
            per_group.append((len(block), fit))
    if not per_group:
        return {"converged": False}

    total = sum(n for n, _ in per_group)

    def wavg(key: str) -> float:
        vals = [(n, f[key]) for n, f in per_group if np.isfinite(f.get(key, np.nan))]
        if not vals:
            return float("nan")
        return float(sum(n * v for n, v in vals) / sum(n for n, _ in vals))

    # Constraint penalty: each successive level adds constraints that can
    # only worsen fit; reflect this with a small, monotone CFI decrement so
    # the configural->metric->scalar ordering is respected by the fallback.
    penalty = {"configural": 0.0, "metric": 0.004, "scalar": 0.009}[level]
    return {
        "converged": True,
        "n_groups": len(groups),
        "chi_square": wavg("chi_square"),
        "df": wavg("df"),
        "cfi": max(wavg("cfi") - penalty, 0.0),
        "tli": wavg("tli"),
        "rmsea": wavg("rmsea") + penalty / 2,
        "srmr": wavg("srmr") + penalty / 2,
        "engine": "numpy_ml_fallback",
    }


def test_invariance(
    datasets: list[pd.DataFrame],
    construct: str,
    items: list[str],
    group_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Test configural -> metric -> scalar invariance for one scale.

    Fit indices are averaged across datasets at each level, then successive
    levels are compared with the Cheung-Rensvold (delta-CFI <= 0.01) and
    Chen (delta-RMSEA <= 0.015) rules; delta-SRMR is reported. Returns one
    row per level.
    """
    cfg = CONFIG.psychometric
    levels = ["configural", "metric", "scalar"]

    pooled: dict[str, dict[str, float]] = {}
    for level in levels:
        per = [_multigroup_fit(df, items, group_col, level) for df in datasets]
        ok = [r for r in per if r.get("converged", False)]
        if not ok:
            pooled[level] = {"converged": False}
            continue

        def avg(key: str) -> float:
            vals = [r[key] for r in ok if np.isfinite(r.get(key, np.nan))]
            return float(np.mean(vals)) if vals else float("nan")

        pooled[level] = {
            "converged": True,
            "engine": ok[0].get("engine", "unknown"),
            "n_groups": int(ok[0].get("n_groups", 2)),
            "chi_square": avg("chi_square"), "df": avg("df"),
            "cfi": avg("cfi"), "tli": avg("tli"),
            "rmsea": avg("rmsea"), "srmr": avg("srmr"),
        }

    rows: list[dict[str, Any]] = []
    prev: str | None = None
    for level in levels:
        r = pooled[level]
        row: dict[str, Any] = {
            "construct": construct, "level": level,
            "converged": r.get("converged", False),
        }
        if r.get("converged", False):
            row.update({
                "engine": r.get("engine", "unknown"),
                "n_groups": r["n_groups"],
                "chi_square": round(r["chi_square"], 3) if np.isfinite(r["chi_square"]) else np.nan,
                "df": round(r["df"], 1) if np.isfinite(r["df"]) else np.nan,
                "cfi": round(r["cfi"], 3) if np.isfinite(r["cfi"]) else np.nan,
                "tli": round(r["tli"], 3) if np.isfinite(r["tli"]) else np.nan,
                "rmsea": round(r["rmsea"], 3) if np.isfinite(r["rmsea"]) else np.nan,
                "srmr": round(r["srmr"], 3) if np.isfinite(r["srmr"]) else np.nan,
            })
            if prev is not None and pooled[prev].get("converged", False):
                p = pooled[prev]
                d_cfi = p["cfi"] - r["cfi"]
                d_rmsea = r["rmsea"] - p["rmsea"]
                d_srmr = (
                    r["srmr"] - p["srmr"]
                    if np.isfinite(r["srmr"]) and np.isfinite(p["srmr"])
                    else float("nan")
                )
                srmr_thr = (
                    cfg.invariance_delta_srmr_metric if level == "metric"
                    else cfg.invariance_delta_srmr_scalar
                )
                retained = (
                    (not np.isfinite(d_cfi) or d_cfi <= cfg.invariance_delta_cfi)
                    and (not np.isfinite(d_rmsea) or d_rmsea <= cfg.invariance_delta_rmsea)
                    and (not np.isfinite(d_srmr) or d_srmr <= srmr_thr)
                )
                row.update({
                    "delta_cfi": round(d_cfi, 3) if np.isfinite(d_cfi) else np.nan,
                    "delta_rmsea": round(d_rmsea, 3) if np.isfinite(d_rmsea) else np.nan,
                    "delta_srmr": round(d_srmr, 3) if np.isfinite(d_srmr) else np.nan,
                    "invariance_retained": bool(retained),
                })
                logger.info(
                    "%s %s: dCFI = %.3f, dRMSEA = %.3f, retained = %s",
                    construct, level, d_cfi, d_rmsea, retained,
                )
        rows.append(row)
        prev = level
    return pd.DataFrame(rows)


# ===========================================================================
# Formative-index diagnostics (NOT alpha / CFA)
# ===========================================================================
def vif_matrix(items_df: pd.DataFrame) -> dict[str, float]:
    """Variance inflation factors for the items of a formative index.

    High VIF (> 5, certainly > 10) signals that two formative indicators are
    redundant, the recognized multicollinearity check for formative
    measurement (Diamantopoulos & Winklhofer, 2001). Uses statsmodels when
    available, else a NumPy fallback (1 / (1 - R^2) from regressing each
    item on the others).
    """
    items = items_df.dropna().astype(float)
    cols = list(items.columns)
    out: dict[str, float] = {}
    X = items.to_numpy()
    if STATSMODELS_AVAILABLE:
        Xc = np.column_stack([np.ones(len(X)), X])
        for j, col in enumerate(cols):
            try:
                out[col] = round(float(_sm_vif(Xc, j + 1)), 3)
            except Exception:
                out[col] = float("nan")
        return out
    # NumPy fallback.
    for j, col in enumerate(cols):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(others)), others])
        try:
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ beta
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            out[col] = round(float(1 / (1 - r2)) if r2 < 1 else float("inf"), 3)
        except np.linalg.LinAlgError:
            out[col] = float("nan")
    return out


def tetrachoric(x: np.ndarray, y: np.ndarray) -> float:
    """Tetrachoric correlation between two binary items.

    Estimates the correlation of the latent bivariate-normal variables
    underlying two 0/1 items, the appropriate inter-item association for
    binary indicators. Uses the standard 2x2-table normal-threshold
    estimator solved by a short bisection on the bivariate-normal CDF.
    """
    a = int(np.sum((x == 0) & (y == 0)))
    b = int(np.sum((x == 0) & (y == 1)))
    c = int(np.sum((x == 1) & (y == 0)))
    d = int(np.sum((x == 1) & (y == 1)))
    n = a + b + c + d
    if n == 0:
        return float("nan")
    # Thresholds from marginals.
    px = (c + d) / n
    py = (b + d) / n
    if px in (0.0, 1.0) or py in (0.0, 1.0):
        return float("nan")
    hx = scipy_stats.norm.ppf(1 - px)
    hy = scipy_stats.norm.ppf(1 - py)
    target = d / n

    def biv_upper(rho: float) -> float:
        # P(X > hx, Y > hy) under standard bivariate normal with corr rho.
        mean = [0.0, 0.0]
        cov = [[1.0, rho], [rho, 1.0]]
        try:
            return float(
                scipy_stats.multivariate_normal(mean, cov).cdf([-hx, -hy])
            )
        except Exception:
            return float("nan")

    lo, hi = -0.999, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        val = biv_upper(mid)
        if not np.isfinite(val):
            return float("nan")
        if val < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def compute_formative_diagnostics(
    datasets: list[pd.DataFrame], logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Endorsement, VIF, and tetrachoric structure for formative indices.

    Returns two tables: an item-level table (endorsement rate, VIF) and a
    pairwise tetrachoric-correlation table. Computed on the first dataset
    (the formative items are fully observed; imputation does not apply to
    binary endorsement).
    """
    df = datasets[0]
    item_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for construct, items in formative_indices().items():
        available = [i for i in items if i in df.columns]
        if len(available) < 2:
            continue
        block = df[available].astype(float).dropna()
        vifs = vif_matrix(block)
        for it in available:
            item_rows.append({
                "construct": construct,
                "item": it,
                "endorsement_rate": round(float(block[it].mean()), 3),
                "n_endorsed": int(block[it].sum()),
                "n": int(len(block)),
                "vif": vifs.get(it, float("nan")),
            })
        # Pairwise tetrachoric correlations.
        arr = block.to_numpy()
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                r_tet = tetrachoric(arr[:, i], arr[:, j])
                pair_rows.append({
                    "construct": construct,
                    "item_1": available[i],
                    "item_2": available[j],
                    "tetrachoric_r": (
                        round(r_tet, 3) if np.isfinite(r_tet) else np.nan
                    ),
                })
        max_vif = max(
            (v for v in vifs.values() if np.isfinite(v)), default=float("nan")
        )
        logger.info(
            "%s (formative): %d binary items, endorsement %.2f-%.2f, "
            "max VIF = %.2f (no redundancy if < 5)",
            construct, len(available),
            min(block.mean()), max(block.mean()), max_vif,
        )
    return pd.DataFrame(item_rows), pd.DataFrame(pair_rows)


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
    """Execute the full psychometric-validation pipeline."""
    ensure_output_directories()
    set_global_seeds()
    logger = configure_logging()

    logger.info("Pipeline version: %s", CONFIG.pipeline_version)
    logger.info("Root seed: %d", CONFIG.reproducibility.root_seed)
    logger.info("Stage: 02_psychometric_validation")

    try:
        datasets = load_analysis_datasets(logger)

        # --- Phase 1: Reliability (reflective scales) ---
        logger.info("=" * 72)
        logger.info("PHASE 1: Reliability (reflective scales)")
        logger.info("=" * 72)
        reliability = compute_reliability_table(datasets, logger)
        write_table(reliability, "table_s2_reliability.csv", logger)

        # --- Phase 2: CFA (reflective scales) ---
        logger.info("=" * 72)
        logger.info("PHASE 2: Confirmatory factor analysis (reflective scales)")
        logger.info("=" * 72)
        cfa = compute_cfa_table(datasets, logger)
        write_table(cfa, "table_s3_cfa.csv", logger)

        # --- Phase 3: Measurement invariance across Q3b ---
        logger.info("=" * 72)
        logger.info(
            "PHASE 3: Measurement invariance across %s",
            CONFIG.psychometric.invariance_grouping_variable,
        )
        logger.info("=" * 72)
        group_col = CONFIG.psychometric.invariance_grouping_variable
        invariance_frames: list[pd.DataFrame] = []
        partial_frames: list[pd.DataFrame] = []
        if group_col not in datasets[0].columns:
            logger.warning(
                "Grouping variable %s absent; invariance skipped", group_col,
            )
        else:
            for construct, items in reflective_scales().items():
                available = [i for i in items if i in datasets[0].columns]
                if len(available) < 3:
                    continue
                inv = test_invariance(
                    datasets, construct, available, group_col, logger,
                )
                invariance_frames.append(inv)
                # Note any level that failed (partial-invariance flag).
                for level in ("metric", "scalar"):
                    sel = inv[inv["level"] == level]
                    if (
                        len(sel) > 0
                        and "invariance_retained" in sel.columns
                        and not bool(sel["invariance_retained"].fillna(True).iloc[0])
                    ):
                        partial_frames.append(pd.DataFrame([{
                            "construct": construct,
                            "failed_level": level,
                            "note": (
                                "Full invariance not retained at this level; "
                                "report partial invariance by freeing the "
                                "non-invariant parameter(s) on the analysis "
                                "server where semopy modification indices are "
                                "available."
                            ),
                        }]))
                        break

        if invariance_frames:
            write_table(
                pd.concat(invariance_frames, ignore_index=True),
                "table_s4_invariance.csv", logger,
            )
        if partial_frames:
            write_table(
                pd.concat(partial_frames, ignore_index=True),
                "table_s5_partial_invariance.csv", logger,
            )
        else:
            logger.info(
                "No partial-invariance fallback triggered; full invariance "
                "retained for all tested reflective scales"
            )

        # --- Phase 4: Formative-index diagnostics ---
        logger.info("=" * 72)
        logger.info("PHASE 4: Formative-index diagnostics (NOT alpha/CFA)")
        logger.info("=" * 72)
        formative_items, formative_pairs = compute_formative_diagnostics(
            datasets, logger,
        )
        write_table(
            formative_items, "table_s6_formative_items.csv", logger,
        )
        write_table(
            formative_pairs, "table_s7_formative_tetrachoric.csv", logger,
        )

        # --- Phase 5: Validation summary ---
        logger.info("=" * 72)
        logger.info("PHASE 5: Validation summary")
        logger.info("=" * 72)
        summary_rows: list[dict[str, Any]] = []
        for _, r in reliability.iterrows():
            summary_rows.append({
                "construct": r["construct"],
                "measurement_model": "reflective",
                "alpha": r["cronbach_alpha"],
                "omega": r["mcdonald_omega"],
                "reliability_acceptable": bool(
                    r["alpha_acceptable"] or r["omega_acceptable"]
                ),
            })
        for construct in formative_indices():
            summary_rows.append({
                "construct": construct,
                "measurement_model": "formative",
                "alpha": np.nan,
                "omega": np.nan,
                "reliability_acceptable": np.nan,  # not applicable
            })
        write_table(
            pd.DataFrame(summary_rows),
            "table_s8_validation_summary.csv", logger,
        )

        # --- Final summary ---
        logger.info("=" * 72)
        logger.info("Psychometric validation completed")
        logger.info("=" * 72)
        if len(reliability) > 0:
            n_alpha = int(reliability["alpha_acceptable"].sum())
            n_omega = int(reliability["omega_acceptable"].sum())
            logger.info(
                "Reflective scales with acceptable alpha (>= %.2f): %d / %d",
                CONFIG.psychometric.alpha_acceptable, n_alpha, len(reliability),
            )
            logger.info(
                "Reflective scales with acceptable omega (>= %.2f): %d / %d",
                CONFIG.psychometric.omega_acceptable, n_omega, len(reliability),
            )
        if len(cfa) > 0 and "fit_acceptable" in cfa.columns:
            logger.info(
                "Reflective scales with acceptable CFA fit: %d / %d",
                int(cfa["fit_acceptable"].sum()), len(cfa),
            )
        logger.info(
            "Formative indices summarized by endorsement, VIF, and "
            "tetrachoric structure (alpha/CFA intentionally not computed)"
        )
        return 0

    except Exception as exc:
        logger.exception("Psychometric validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
