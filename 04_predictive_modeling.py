"""
04_predictive_modeling.py
=========================

Predictive modeling and AI-moderation pipeline (Stage 4 of 7) for the
manuscript "Individual Differences in Doctoral Learning Adaptation and
Well-Being: Academic Pressure, Supervisor Support, Career Uncertainty, and
the Moderating Role of Generative AI among Chinese PhD Students."

This stage implements the principal novel contribution of the paper:
testing whether generative-AI engagement (use, comfort, and concerns)
moderates the relationships between the three focal predictors (academic
pressure, supervisor support, career uncertainty) and well-being.

Two complementary methods are used, and they answer different questions:

  * CLASSICAL OLS MODERATION estimates the linear interaction of each
    AI moderator with each focal predictor, with bootstrap confidence
    intervals, simple slopes at conditional moderator values, and a
    Benjamini-Hochberg false-discovery-rate correction across the family
    of nine interaction tests. This is the confirmatory, interpretable
    analysis (Hayes, 2018; Aiken & West, 1991).

  * GRADIENT-BOOSTING WITH SHAP detects NON-LINEAR moderation that the
    linear product term cannot capture, and ranks predictor and
    interaction importance through the SHAP interaction-value
    decomposition (Lundberg & Lee, 2017). This is the exploratory,
    pattern-discovery analysis.

A note on statistical power
---------------------------
The binary AI-use moderator is highly imbalanced on the published data
(39 users vs. 361 non-users). Interaction tests involving ai_use are
therefore low-powered, and the script logs this explicitly so that null
interactions for ai_use are not over-interpreted. The continuous
moderators (ai_comfort, ai_concerns) do not have this limitation.

Because Script 01 produces a single analysis dataset (multiple imputation
is disabled: the published data have no missingness), estimates are not
pooled across imputations; the bootstrap is the sole source of interval
estimates. If a future wave enables imputation, the per-imputation results
would be combined by Rubin's rules, but that path is out of scope here.

Dependency handling
-------------------
OLS uses statsmodels when available, otherwise a NumPy ordinary-least-
squares implementation with analytic standard errors and a parametric
covariance for simple slopes. Gradient boosting uses XGBoost when
available, otherwise scikit-learn's GradientBoostingRegressor. SHAP
interaction values use the shap package when available, otherwise a
model-agnostic partial-dependence-based interaction strength (Friedman's
H-statistic family) computed from the fitted ensemble. The engine actually
used is recorded in every output table.

Methodological references
-------------------------
Aiken, L. S., & West, S. G. (1991). Multiple regression: Testing and
    interpreting interactions. Sage.
Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate. JRSS-B, 57(1), 289-300.
Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule
    ensembles. Annals of Applied Statistics, 2(3), 916-954.
Hayes, A. F. (2018). Introduction to mediation, moderation, and
    conditional process analysis (2nd ed.). Guilford Press.
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
    model predictions. NeurIPS 2017.

Author: BEFOUM Stephane Richard
Target journal: Learning and Individual Differences (Elsevier)
"""

from __future__ import annotations

import logging
import sys
import warnings
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as scipy_stats

from configs import CONFIG, ensure_output_directories, set_global_seeds

# OLS engine: statsmodels preferred, NumPy fallback.
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Gradient-boosting engine: XGBoost preferred, sklearn fallback.
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# SHAP for interaction values.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# Construct definitions
# ===========================================================================
FOCAL_PREDICTORS: tuple[str, ...] = (
    "academic_pressure",
    "supervisor_support",
    "career_uncertainty",
)
OUTCOME: str = "wellbeing"
AI_MODERATORS: tuple[str, ...] = ("ai_use", "ai_comfort", "ai_concerns")
COVARIATES: tuple[str, ...] = ("Q3a", "Q3b")
BINARY_MODERATORS: frozenset[str] = frozenset({"ai_use"})


# ===========================================================================
# Logging configuration
# ===========================================================================
def configure_logging() -> logging.Logger:
    """Configure structured logging to stdout and a timestamped file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = CONFIG.paths.logs_dir / f"04_predictive_{timestamp}.log"

    logger = logging.getLogger("predictive_modeling")
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
        "Engines: OLS = %s, boosting = %s, SHAP = %s",
        "statsmodels" if STATSMODELS_AVAILABLE else "numpy_fallback",
        "xgboost" if XGBOOST_AVAILABLE else "sklearn_fallback",
        "shap" if SHAP_AVAILABLE else "h_statistic_fallback",
    )
    return logger


# ===========================================================================
# Data loading
# ===========================================================================
def load_analysis_dataset(logger: logging.Logger) -> pd.DataFrame:
    """Load the canonical analysis dataset produced by Script 01."""
    if CONFIG.imputation.enable:
        path = CONFIG.paths.imputed_path(1)
        logger.warning(
            "Imputation enabled; using imputation 1 as the reference for "
            "moderation. Rubin's-rules pooling is not performed here.",
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


# ===========================================================================
# OLS fitting (statsmodels or NumPy), returning everything simple slopes need
# ===========================================================================
def mean_center(series: pd.Series) -> pd.Series:
    """Mean-center a continuous variable for product-term computation."""
    return series - series.mean()


def _numpy_ols(
    y: np.ndarray, X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Ordinary least squares with analytic covariance.

    Returns coefficients, standard errors, the full coefficient covariance
    matrix, and R-squared. The covariance is sigma^2 (X'X)^-1 with the
    usual unbiased sigma^2 = RSS / (n - k).
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    rss = float(resid @ resid)
    sigma2 = rss / (n - k) if n > k else float("nan")
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    return beta, se, cov, r2


def fit_moderation(
    df: pd.DataFrame,
    predictor: str,
    moderator: str,
    covariates: list[str],
) -> dict[str, float] | None:
    """Fit one moderation regression and return coefficients + covariance.

    The model is y = b0 + b_x*x + b_m*m + b_xm*(x*m) + controls, where x is
    the mean-centered predictor and m is the mean-centered continuous
    moderator (or the raw 0/1 value for the binary AI-use moderator). The
    returned dict carries the interaction estimate, its p-value, R-squared,
    and the covariance entries needed to compute simple-slope standard
    errors downstream. Returns None if the model cannot be fit.
    """
    required = [predictor, moderator, OUTCOME] + covariates
    work = df[required].dropna().copy()
    if len(work) < 100:
        return None

    work["_x"] = mean_center(work[predictor])
    if moderator in BINARY_MODERATORS:
        work["_m"] = work[moderator].astype(float)
        m_grand_mean = 0.0
        m_sd = 1.0
    else:
        work["_m"] = mean_center(work[moderator])
        m_grand_mean = float(df[moderator].mean())
        m_sd = float(df[moderator].std())
    work["_xm"] = work["_x"] * work["_m"]

    design = ["_x", "_m", "_xm"] + covariates

    if STATSMODELS_AVAILABLE:
        try:
            X = sm.add_constant(work[design])
            model = sm.OLS(work[OUTCOME], X).fit()
            cov = model.cov_params()
            return {
                "n": int(len(work)),
                "b_x": float(model.params["_x"]),
                "b_m": float(model.params["_m"]),
                "b_xm": float(model.params["_xm"]),
                "se_xm": float(model.bse["_xm"]),
                "p_xm": float(model.pvalues["_xm"]),
                "r_squared": float(model.rsquared),
                "cov_x_x": float(cov.loc["_x", "_x"]),
                "cov_x_xm": float(cov.loc["_x", "_xm"]),
                "cov_xm_xm": float(cov.loc["_xm", "_xm"]),
                "moderator_grand_mean": m_grand_mean,
                "moderator_sd": m_sd,
                "engine": "statsmodels",
            }
        except Exception:
            return None

    # NumPy fallback.
    y = work[OUTCOME].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(work))] + [work[c].to_numpy(dtype=float) for c in design])
    try:
        beta, se, cov, r2 = _numpy_ols(y, X)
    except np.linalg.LinAlgError:
        return None
    # Column order: const, _x, _m, _xm, *covariates.
    idx = {name: i for i, name in enumerate(["const"] + design)}
    se_xm = float(se[idx["_xm"]])
    b_xm = float(beta[idx["_xm"]])
    n, k = X.shape
    t = b_xm / se_xm if se_xm > 0 else float("nan")
    p_xm = float(2 * scipy_stats.t.sf(abs(t), n - k)) if np.isfinite(t) else float("nan")
    return {
        "n": int(len(work)),
        "b_x": float(beta[idx["_x"]]),
        "b_m": float(beta[idx["_m"]]),
        "b_xm": b_xm,
        "se_xm": se_xm,
        "p_xm": p_xm,
        "r_squared": float(r2),
        "cov_x_x": float(cov[idx["_x"], idx["_x"]]),
        "cov_x_xm": float(cov[idx["_x"], idx["_xm"]]),
        "cov_xm_xm": float(cov[idx["_xm"], idx["_xm"]]),
        "moderator_grand_mean": m_grand_mean,
        "moderator_sd": m_sd,
        "engine": "numpy_fallback",
    }


# ===========================================================================
# Bootstrap confidence interval for the interaction coefficient
# ===========================================================================
def bootstrap_interaction(
    df: pd.DataFrame,
    predictor: str,
    moderator: str,
    covariates: list[str],
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the interaction coefficient.

    Resamples respondents with replacement and refits the moderation model
    on each resample, collecting the interaction coefficient. Returns the
    (lower, upper) percentile bounds; NaN bounds if too few resamples
    succeed.
    """
    required = [predictor, moderator, OUTCOME] + covariates
    work = df[required].dropna().copy()
    if len(work) < 100:
        return float("nan"), float("nan")

    work["_x"] = mean_center(work[predictor])
    if moderator in BINARY_MODERATORS:
        work["_m"] = work[moderator].astype(float)
    else:
        work["_m"] = mean_center(work[moderator])
    work["_xm"] = work["_x"] * work["_m"]
    design = ["_x", "_m", "_xm"] + covariates

    y_full = work[OUTCOME].to_numpy(dtype=float)
    X_full = np.column_stack(
        [np.ones(len(work))] + [work[c].to_numpy(dtype=float) for c in design]
    )
    xm_col = 3  # const, _x, _m, _xm
    n = len(work)
    rng = np.random.default_rng(random_state)

    def _iteration(seed: int) -> float:
        local = np.random.default_rng(seed)
        idx = local.integers(0, n, size=n)
        Xb, yb = X_full[idx], y_full[idx]
        try:
            beta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ yb
            return float(beta[xm_col])
        except np.linalg.LinAlgError:
            return float("nan")

    seeds = rng.integers(0, 1_000_000, size=n_bootstrap)
    coefs = Parallel(
        n_jobs=CONFIG.hardware.n_cpu_workers,
        backend=CONFIG.hardware.parallel_backend,
    )(delayed(_iteration)(int(s)) for s in seeds)
    valid = [c for c in coefs if not np.isnan(c)]

    if len(valid) < max(100, n_bootstrap * 0.5):
        return float("nan"), float("nan")
    alpha = (1 - CONFIG.predictive.ci_level) / 2
    return (
        float(np.percentile(valid, alpha * 100)),
        float(np.percentile(valid, (1 - alpha) * 100)),
    )


# ===========================================================================
# Benjamini-Hochberg FDR
# ===========================================================================
def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (monotone, clipped to 1)."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


# ===========================================================================
# Classical moderation battery
# ===========================================================================
def run_classical_moderation(
    df: pd.DataFrame, logger: logging.Logger,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run all predictor x moderator interaction tests with bootstrap CIs.

    Returns the results table (one row per interaction, with FDR-adjusted
    p-values) and the per-interaction raw fits used by simple-slope
    analysis.
    """
    cfg = CONFIG.predictive
    covariates = [c for c in COVARIATES if c in df.columns]
    base_seed = (
        CONFIG.reproducibility.root_seed
        + CONFIG.reproducibility.predictive_seed_offset
    )

    combos = list(product(FOCAL_PREDICTORS, AI_MODERATORS))
    logger.info(
        "Testing %d predictor x moderator interactions on well-being "
        "(covariates: %s)", len(combos), covariates,
    )

    rows: list[dict[str, Any]] = []
    raw_fits: list[dict[str, Any]] = []

    for i, (predictor, moderator) in enumerate(combos):
        fit = fit_moderation(df, predictor, moderator, covariates)
        if fit is None:
            logger.warning(
                "%s x %s: model could not be fit; skipped", predictor, moderator,
            )
            continue

        ci_lo, ci_hi = bootstrap_interaction(
            df, predictor, moderator, covariates,
            n_bootstrap=cfg.n_bootstrap_ci,
            random_state=base_seed + 1000 * i,
        )

        low_power = moderator in BINARY_MODERATORS
        rows.append({
            "predictor": predictor,
            "moderator": moderator,
            "outcome": OUTCOME,
            "n": fit["n"],
            "interaction_coefficient": round(fit["b_xm"], 4),
            "interaction_se": round(fit["se_xm"], 4),
            "interaction_p": round(fit["p_xm"], 4) if np.isfinite(fit["p_xm"]) else np.nan,
            "ci_lower_bootstrap": round(ci_lo, 4) if np.isfinite(ci_lo) else np.nan,
            "ci_upper_bootstrap": round(ci_hi, 4) if np.isfinite(ci_hi) else np.nan,
            "r_squared": round(fit["r_squared"], 4),
            "significant_p": bool(np.isfinite(fit["p_xm"]) and fit["p_xm"] < 0.05),
            "significant_bootstrap": bool(
                np.isfinite(ci_lo) and np.isfinite(ci_hi)
                and (ci_lo > 0 or ci_hi < 0)
            ),
            "low_power_flag": low_power,
            "engine": fit["engine"],
        })
        raw_fits.append({
            "predictor": predictor, "moderator": moderator, "fit": fit,
        })
        logger.info(
            "%s x %s: b = %.4f, p = %.4f, boot CI = [%.4f, %.4f]%s",
            predictor, moderator, fit["b_xm"], fit["p_xm"], ci_lo, ci_hi,
            "  [LOW POWER: binary moderator]" if low_power else "",
        )

    results = pd.DataFrame(rows)

    # Benjamini-Hochberg across the family of interaction tests.
    if len(results) > 0 and results["interaction_p"].notna().any():
        mask = results["interaction_p"].notna()
        fdr = benjamini_hochberg(results.loc[mask, "interaction_p"].to_numpy())
        results["interaction_p_fdr"] = np.nan
        results.loc[mask, "interaction_p_fdr"] = np.round(fdr, 4)
        results["significant_fdr"] = results["interaction_p_fdr"] < 0.05

    return results, raw_fits


# ===========================================================================
# Simple slopes for significant interactions
# ===========================================================================
def compute_simple_slopes(
    raw_fit: dict[str, Any],
) -> pd.DataFrame:
    """Conditional slopes of the predictor at low/mean/high moderator values.

    Uses Var(slope | m=w) = Var(b_x) + 2w*Cov(b_x, b_xm) + w^2*Var(b_xm),
    where w is the moderator value expressed as a deviation from the grand
    mean (Aiken & West, 1991). For the binary moderator, slopes are
    reported at the two levels (non-user, user).
    """
    fit = raw_fit["fit"]
    moderator = raw_fit["moderator"]
    b_x, b_xm = fit["b_x"], fit["b_xm"]
    cov_xx, cov_xxm, cov_xmxm = fit["cov_x_x"], fit["cov_x_xm"], fit["cov_xm_xm"]
    m_mean, m_sd = fit["moderator_grand_mean"], fit["moderator_sd"]

    if moderator in BINARY_MODERATORS:
        levels = {"non_user": 0.0, "user": 1.0}
    else:
        levels = {
            "low_minus_1sd": m_mean - m_sd,
            "mean": m_mean,
            "high_plus_1sd": m_mean + m_sd,
        }

    rows: list[dict[str, Any]] = []
    for label, value in levels.items():
        w = value if moderator in BINARY_MODERATORS else value - m_mean
        slope = b_x + b_xm * w
        var = cov_xx + 2 * w * cov_xxm + (w ** 2) * cov_xmxm
        se = float(np.sqrt(max(var, 0.0)))
        t = slope / se if se > 0 else float("nan")
        p = float(2 * scipy_stats.norm.sf(abs(t))) if np.isfinite(t) else float("nan")
        rows.append({
            "predictor": raw_fit["predictor"],
            "moderator": moderator,
            "moderator_level": label,
            "moderator_value": round(float(value), 3),
            "simple_slope": round(float(slope), 4),
            "se": round(se, 4),
            "t": round(float(t), 3) if np.isfinite(t) else np.nan,
            "p_value": round(p, 4) if np.isfinite(p) else np.nan,
            "significant": bool(np.isfinite(p) and p < 0.05),
        })
    return pd.DataFrame(rows)


def run_simple_slopes(
    results: pd.DataFrame,
    raw_fits: list[dict[str, Any]],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compute simple slopes for every interaction flagged significant.

    An interaction is probed if it is significant by either the parametric
    p-value or the bootstrap CI. If none is significant, an empty frame is
    returned and that is reported (a legitimate, common outcome).
    """
    if len(results) == 0:
        return pd.DataFrame()
    sig = results[
        results.get("significant_p", False) | results.get("significant_bootstrap", False)
    ]
    logger.info("Interactions flagged significant to probe: %d", len(sig))
    if len(sig) == 0:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for _, row in sig.iterrows():
        match = next(
            (r for r in raw_fits
             if r["predictor"] == row["predictor"]
             and r["moderator"] == row["moderator"]),
            None,
        )
        if match is None:
            continue
        frames.append(compute_simple_slopes(match))
        logger.info(
            "Simple slopes computed for %s x %s",
            row["predictor"], row["moderator"],
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ===========================================================================
# Gradient-boosting model (XGBoost or sklearn)
# ===========================================================================
def build_booster(random_state: int):
    """Construct a gradient-boosting regressor with the configured settings.

    Returns an XGBoost regressor when XGBoost is installed, otherwise a
    scikit-learn GradientBoostingRegressor parameterized to match as
    closely as the two libraries allow.
    """
    cfg = CONFIG.predictive
    if XGBOOST_AVAILABLE:
        return xgb.XGBRegressor(
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            n_estimators=cfg.xgb_n_estimators,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            random_state=random_state,
            verbosity=0,
            tree_method="hist",
        )
    return GradientBoostingRegressor(
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        n_estimators=cfg.xgb_n_estimators,
        subsample=cfg.xgb_subsample,
        max_features=cfg.xgb_colsample_bytree,
        random_state=random_state,
    )


def train_and_evaluate_booster(
    df: pd.DataFrame, features: list[str], logger: logging.Logger,
) -> tuple[Any, np.ndarray, dict[str, float]]:
    """Train the booster with repeated CV and return it with CV R-squared."""
    cfg = CONFIG.predictive
    work = df[features + [OUTCOME]].dropna().reset_index(drop=True)
    X = work[features].to_numpy(dtype=float)
    y = work[OUTCOME].to_numpy(dtype=float)
    seed = (
        CONFIG.reproducibility.root_seed
        + CONFIG.reproducibility.predictive_seed_offset
    )

    cv = RepeatedKFold(
        n_splits=cfg.cv_n_folds, n_repeats=cfg.cv_n_repeats, random_state=seed,
    )
    r2_scores: list[float] = []
    for fold, (tr, te) in enumerate(cv.split(X)):
        model = build_booster(seed + fold)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        ss_res = float(np.sum((y[te] - pred) ** 2))
        ss_tot = float(np.sum((y[te] - y[te].mean()) ** 2))
        r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))

    final = build_booster(seed)
    final.fit(X, y)
    meta = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv_r2_mean": round(float(np.nanmean(r2_scores)), 4),
        "cv_r2_sd": round(float(np.nanstd(r2_scores, ddof=1)), 4),
        "engine": "xgboost" if XGBOOST_AVAILABLE else "sklearn_gbr",
    }
    logger.info(
        "Booster (%s): CV R^2 = %.4f (SD = %.4f) over %d x %d folds",
        meta["engine"], meta["cv_r2_mean"], meta["cv_r2_sd"],
        cfg.cv_n_folds, cfg.cv_n_repeats,
    )
    return final, X, meta


# ===========================================================================
# Interaction importance: SHAP or H-statistic fallback
# ===========================================================================
def shap_interactions(
    model: Any, X: np.ndarray, features: list[str], logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main-effect and pairwise interaction importances via SHAP.

    Returns a main-effects table and a pairwise-interaction table. Uses the
    shap TreeExplainer when available. The off-diagonal SHAP interaction
    entries are doubled (SHAP splits a pair's interaction symmetrically).
    """
    explainer = shap.TreeExplainer(model)
    inter = explainer.shap_interaction_values(X)
    mean_abs = np.abs(inter).mean(axis=0)

    main_rows = [
        {"feature": features[i], "mean_abs_shap_main": float(mean_abs[i, i])}
        for i in range(len(features))
    ]
    pair_rows = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pair_rows.append({
                "feature_1": features[i],
                "feature_2": features[j],
                "mean_abs_shap_interaction": float(2 * mean_abs[i, j]),
            })
    return (
        pd.DataFrame(main_rows).sort_values(
            "mean_abs_shap_main", ascending=False).reset_index(drop=True),
        pd.DataFrame(pair_rows).sort_values(
            "mean_abs_shap_interaction", ascending=False).reset_index(drop=True),
    )


def h_statistic_interactions(
    model: Any, X: np.ndarray, features: list[str], logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model-agnostic interaction strength via a partial-dependence H-statistic.

    For each pair (j, k), Friedman's H-statistic compares the joint partial
    dependence to the sum of the two univariate partial dependences; a large
    value indicates the model's prediction depends on j and k jointly beyond
    their additive contributions. Main-effect importance is the variance of
    each univariate partial-dependence function. This is the fallback used
    when the shap package is unavailable; it requires no extra dependency
    and captures the same non-additive structure SHAP interaction values
    summarize.
    """
    rng = np.random.default_rng(
        CONFIG.reproducibility.root_seed
        + CONFIG.reproducibility.predictive_seed_offset
    )
    n = X.shape[0]
    grid_n = min(20, n)
    sample_idx = rng.choice(n, size=min(150, n), replace=False)
    Xs = X[sample_idx]

    def partial_dependence_1d(j: int, grid: np.ndarray) -> np.ndarray:
        out = np.empty(len(grid))
        Xtmp = Xs.copy()
        for g, v in enumerate(grid):
            Xtmp[:, j] = v
            out[g] = float(model.predict(Xtmp).mean())
        return out

    def partial_dependence_2d(j: int, k: int, gj: np.ndarray, gk: np.ndarray) -> np.ndarray:
        out = np.empty((len(gj), len(gk)))
        Xtmp = Xs.copy()
        for a, vj in enumerate(gj):
            for b, vk in enumerate(gk):
                Xtmp[:, j] = vj
                Xtmp[:, k] = vk
                out[a, b] = float(model.predict(Xtmp).mean())
        return out

    grids = {
        j: np.quantile(X[:, j], np.linspace(0.05, 0.95, grid_n))
        for j in range(len(features))
    }
    pd1 = {j: partial_dependence_1d(j, grids[j]) for j in range(len(features))}
    pd1_centered = {j: pd1[j] - pd1[j].mean() for j in pd1}

    main_rows = [
        {"feature": features[j], "pd_variance_main": float(np.var(pd1[j]))}
        for j in range(len(features))
    ]

    pair_rows = []
    for j in range(len(features)):
        for k in range(j + 1, len(features)):
            pd2 = partial_dependence_2d(j, k, grids[j], grids[k])
            pd2_centered = pd2 - pd2.mean()
            additive = pd1_centered[j][:, None] + pd1_centered[k][None, :]
            numerator = float(np.sum((pd2_centered - additive) ** 2))
            denominator = float(np.sum(pd2_centered ** 2))
            h2 = numerator / denominator if denominator > 0 else 0.0
            pair_rows.append({
                "feature_1": features[j],
                "feature_2": features[k],
                "h_statistic": round(float(np.sqrt(max(h2, 0.0))), 4),
            })

    return (
        pd.DataFrame(main_rows).sort_values(
            "pd_variance_main", ascending=False).reset_index(drop=True),
        pd.DataFrame(pair_rows).sort_values(
            "h_statistic", ascending=False).reset_index(drop=True),
    )


def run_ml_moderation(
    df: pd.DataFrame, logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Train the booster and compute interaction importances.

    Features are the three focal predictors plus the three AI moderators.
    Returns performance, main-effect, and interaction tables. The
    interaction table uses SHAP when available, otherwise the H-statistic
    fallback; the column names differ accordingly and the engine is
    recorded.
    """
    features = list(FOCAL_PREDICTORS) + list(AI_MODERATORS)
    available = [f for f in features if f in df.columns]
    if len(available) < len(features):
        logger.warning(
            "Missing ML features: %s", set(features) - set(available),
        )

    model, X, meta = train_and_evaluate_booster(df, available, logger)
    performance = pd.DataFrame([meta])

    if SHAP_AVAILABLE:
        try:
            main, inter = shap_interactions(model, X, available, logger)
            main["engine"] = "shap"
            inter["engine"] = "shap"
            logger.info("Interaction importances computed via SHAP")
            return {"performance": performance, "main_effects": main, "interactions": inter}
        except Exception as exc:
            logger.warning("SHAP failed (%s); using H-statistic fallback", exc)

    main, inter = h_statistic_interactions(model, X, available, logger)
    main["engine"] = "h_statistic"
    inter["engine"] = "h_statistic"
    logger.info("Interaction importances computed via H-statistic fallback")
    return {"performance": performance, "main_effects": main, "interactions": inter}


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
    """Execute the full predictive-modeling and AI-moderation pipeline."""
    ensure_output_directories()
    set_global_seeds()
    logger = configure_logging()

    logger.info("Pipeline version: %s", CONFIG.pipeline_version)
    logger.info("Root seed: %d", CONFIG.reproducibility.root_seed)
    logger.info("Bootstrap iterations per interaction: %d",
                CONFIG.predictive.n_bootstrap_ci)
    logger.info("Stage: 04_predictive_modeling")

    try:
        df = load_analysis_dataset(logger)
        for var in list(FOCAL_PREDICTORS) + list(AI_MODERATORS) + [OUTCOME]:
            if var not in df.columns:
                raise ValueError(
                    f"Required variable {var} not found. Verify Script 01 "
                    f"produced all construct scores."
                )

        # Report the AI-use imbalance up front.
        if "ai_use" in df.columns:
            counts = df["ai_use"].value_counts().to_dict()
            logger.info(
                "AI-use distribution: %s. Interactions with ai_use are "
                "low-powered and flagged accordingly.", counts,
            )

        # --- Phase 1: Classical OLS moderation ---
        logger.info("=" * 72)
        logger.info("PHASE 1: Classical OLS moderation with bootstrap CIs")
        logger.info("=" * 72)
        classical, raw_fits = run_classical_moderation(df, logger)
        write_table(classical, "table_s14_classical_moderation.csv", logger)

        # --- Phase 2: Simple slopes ---
        logger.info("=" * 72)
        logger.info("PHASE 2: Simple slopes for significant interactions")
        logger.info("=" * 72)
        slopes = run_simple_slopes(classical, raw_fits, logger)
        if len(slopes) > 0:
            write_table(slopes, "table_s15_simple_slopes.csv", logger)
        else:
            logger.info(
                "No interaction reached significance; no simple slopes to "
                "report (a legitimate null result)."
            )

        # --- Phase 3: Gradient boosting with interaction importances ---
        logger.info("=" * 72)
        logger.info("PHASE 3: Gradient boosting with interaction importances")
        logger.info("=" * 72)
        ml = run_ml_moderation(df, logger)
        write_table(ml["performance"], "table_s16_ml_performance.csv", logger)
        write_table(ml["main_effects"], "table_s17_ml_main_effects.csv", logger)
        write_table(ml["interactions"], "table_s18_ml_interactions.csv", logger)

        # --- Phase 4: Diagnostics summary ---
        logger.info("=" * 72)
        logger.info("PHASE 4: Diagnostics summary")
        logger.info("=" * 72)
        diag_rows = [
            {
                "stage": "classical_moderation",
                "method": "OLS interaction + percentile bootstrap + BH-FDR",
                "engine": classical["engine"].iloc[0] if len(classical) else "n/a",
                "n_tests": len(classical),
                "n_significant_p": int(classical["significant_p"].sum()) if len(classical) else 0,
                "n_significant_bootstrap": int(classical["significant_bootstrap"].sum()) if len(classical) else 0,
                "n_significant_fdr": int(classical["significant_fdr"].sum()) if "significant_fdr" in classical.columns else 0,
            },
            {
                "stage": "ml_moderation",
                "method": "gradient boosting + interaction importance",
                "engine": ml["interactions"]["engine"].iloc[0] if len(ml["interactions"]) else "n/a",
                "n_tests": len(ml["interactions"]),
                "n_significant_p": 0,
                "n_significant_bootstrap": 0,
                "n_significant_fdr": 0,
            },
        ]
        write_table(pd.DataFrame(diag_rows), "table_s19_predictive_diagnostics.csv", logger)

        # --- Final summary ---
        logger.info("=" * 72)
        logger.info("Predictive modeling completed")
        logger.info("=" * 72)
        if len(classical) > 0:
            logger.info(
                "Significant interactions: parametric %d, bootstrap %d, "
                "FDR-corrected %d (of %d tests)",
                int(classical["significant_p"].sum()),
                int(classical["significant_bootstrap"].sum()),
                int(classical["significant_fdr"].sum()) if "significant_fdr" in classical.columns else 0,
                len(classical),
            )
        logger.info(
            "Booster CV R^2 = %.4f", ml["performance"]["cv_r2_mean"].iloc[0],
        )
        return 0

    except Exception as exc:
        logger.exception("Predictive modeling failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
