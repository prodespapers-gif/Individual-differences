"""
05_causal_heterogeneity.py
==========================

Causal heterogeneity pipeline (Stage 5 of 7) for the manuscript
"Individual Differences in Doctoral Learning Adaptation and Well-Being:
Academic Pressure, Supervisor Support, Career Uncertainty, and the
Moderating Role of Generative AI among Chinese PhD Students."

This stage estimates the individual treatment effects of supervisor
support on well-being using a causal forest. Where Stage 4 asked whether
AI moderates population-average relationships, this stage asks a different
question: does the effect of supervisor support on well-being vary across
students, and can we characterize who benefits most? The person-specific
conditional average treatment effects (CATEs) complement the
population-average moderation results and operationalize the
heterogeneity-of-adaptation thesis at the individual level.

The treatment is dichotomized supervisor support (at the sample median);
the outcome is well-being; the effect modifiers are the other two focal
constructs plus demographic and workload controls. The analysis is
observational, so the causal interpretation rests on a selection-on-
observables assumption; that assumption is stress-tested explicitly in the
sensitivity stage (Rosenbaum bounds and the E-value).

The pipeline follows the modern causal-forest workflow of Sverdrup,
Petukhova, and Wager (2025), including four elements that an earlier draft
either omitted or implemented ad hoc:

  1. A correct Targeting Operator Characteristic (TOC) curve and its area
     (AUTOC) as defined by Yadlowsky et al. (2024): rank individuals by
     predicted CATE; the TOC at fraction q is the average treatment effect
     among the top-q-ranked individuals minus the overall ATE; AUTOC is the
     area under that curve, with a bootstrap standard error and a
     central-limit-theorem-based test of H0: AUTOC = 0. (The earlier
     draft's cumulative-mean approximation is replaced.)
  2. The Athey-Wager (2019) calibration test ("test_calibration"), which
     regresses a doubly robust score on the mean forest prediction and the
     differential forest prediction; a differential coefficient near 1 with
     a small p-value indicates well-calibrated heterogeneity.
  3. The best linear projection of the CATE on the effect modifiers
     (Semenova & Chernozhukov, 2021), summarizing which baseline
     characteristics drive treatment-effect variation.
  4. A train-test split with quartile stratification for honest
     heterogeneity validation, plus Rosenbaum bounds and the E-value
     (VanderWeele & Ding, 2017) for unmeasured-confounding sensitivity.

Because Script 01 produces a single analysis dataset (imputation disabled),
estimates are not pooled across imputations; honest splitting and
cross-fitting provide valid inference on the single dataset. If a future
wave enables imputation, the per-imputation CATEs would be combined by
Rubin's rules; that path is out of scope here.

Dependency handling
-------------------
The causal forest uses EconML's CausalForestDML when EconML is installed
(the target server). When EconML is unavailable, the script uses a built-in
R-learner causal forest (Nie & Wager, 2021): cross-fitted random-forest
nuisance models (outcome regression and propensity) form the orthogonal
residuals, and a weighted random-forest regression of the R-learner
pseudo-outcome yields the CATE function. This is the same orthogonal,
doubly robust principle EconML implements, so the analysis runs end to end
in either environment; the engine used is recorded in every output table.

Runtime
-------
The forest fit, the AUTOC bootstrap (CONFIG.causal_forest.autoc_n_bootstrap),
and the train-test refit are the compute-intensive steps. With the
camera-ready budget (5000 trees, 500 AUTOC bootstrap replicates) the stage
runs comfortably on the target server; reduce n_trees and autoc_n_bootstrap
for a quick check on a small machine. AUTOC and the calibration test use
cross-fitted (out-of-fold) CATE predictions so the targeting evaluation is
honest rather than optimistic; in-sample CATEs are used only for the
reported per-individual estimates and the best linear projection.

Methodological references
-------------------------
Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests.
    Annals of Statistics, 47(2), 1148-1178.
Athey, S., & Wager, S. (2019). Estimating treatment effects with causal
    forests: An application. Observational Studies, 5(2), 37-51.
Chernozhukov, V., et al. (2018). Double/debiased machine learning. The
    Econometrics Journal, 21(1), C1-C68.
Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous
    treatment effects. Biometrika, 108(2), 299-319.
Rosenbaum, P. R. (2002). Observational studies (2nd ed.). Springer.
Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of
    conditional average treatment effects. The Econometrics Journal,
    24(2), 264-289.
Sverdrup, E., Petukhova, M., & Wager, S. (2025). Estimating treatment
    effect heterogeneity in psychiatry: A review and tutorial with causal
    forests. Int. J. Methods in Psychiatric Research, 34(2), e70015.
VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in
    observational research: Introducing the E-value. Annals of Internal
    Medicine, 167(4), 268-274.
Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., & Wager, S. (2024).
    Evaluating treatment prioritization rules via rank-weighted average
    treatment effects. Journal of the American Statistical Association.

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
from joblib import Parallel, delayed
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split

from configs import CONFIG, ensure_output_directories, set_global_seeds

# EconML is the preferred causal-forest engine.
try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# Logging configuration
# ===========================================================================
def configure_logging() -> logging.Logger:
    """Configure structured logging to stdout and a timestamped file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = CONFIG.paths.logs_dir / f"05_causal_{timestamp}.log"

    logger = logging.getLogger("causal_heterogeneity")
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
        "Causal-forest engine: %s",
        "EconML CausalForestDML" if ECONML_AVAILABLE
        else "built-in R-learner causal forest (Nie & Wager 2021)",
    )
    return logger


# ===========================================================================
# Data loading and design construction
# ===========================================================================
def load_analysis_dataset(logger: logging.Logger) -> pd.DataFrame:
    """Load the canonical analysis dataset produced by Script 01."""
    if CONFIG.imputation.enable:
        path = CONFIG.paths.imputed_path(1)
        logger.warning(
            "Imputation enabled; using imputation 1 as the reference. "
            "Rubin's-rules pooling of CATEs is not performed here.",
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


def construct_treatment(
    df: pd.DataFrame, logger: logging.Logger,
) -> pd.DataFrame:
    """Construct the binary treatment by dichotomizing supervisor support."""
    df = df.copy()
    method = CONFIG.causal_forest.treatment_dichotomization
    if method == "median":
        threshold = float(df["supervisor_support"].median())
    elif method == "mean":
        threshold = float(df["supervisor_support"].mean())
    else:
        raise ValueError(f"Unknown dichotomization method: {method}")

    df["treatment"] = (df["supervisor_support"] >= threshold).astype(int)
    logger.info(
        "Treatment = (supervisor_support >= %.3f): treated = %d, control = %d",
        threshold, int(df["treatment"].sum()), int((1 - df["treatment"]).sum()),
    )
    return df


def select_covariates(df: pd.DataFrame, logger: logging.Logger) -> list[str]:
    """Select effect-modifier / confounder covariates available in the data.

    The set is the two other focal constructs (which are plausibly both
    confounders of the support-wellbeing relationship and effect modifiers)
    plus demographic and workload controls.
    """
    candidates = [
        "academic_pressure", "career_uncertainty", "Q3a", "Q3b", "Q17", "Q18",
    ]
    available = [c for c in candidates if c in df.columns]
    missing = set(candidates) - set(available)
    if missing:
        logger.info("Covariates not available: %s", missing)
    logger.info("Covariates for CATE estimation: %s", available)
    return available


# ===========================================================================
# Causal forest: EconML or built-in R-learner
# ===========================================================================
class RLearnerCausalForest:
    """Built-in R-learner causal forest (Nie & Wager, 2021).

    Cross-fitted random forests estimate the outcome regression m(x) =
    E[Y|X=x] and the propensity e(x) = P(T=1|X=x). The orthogonal residuals
    Y - m(x) and T - e(x) define the R-learner objective, whose minimizer is
    the CATE. The CATE function is fit as a weighted random-forest regression
    of the pseudo-outcome (Y-residual)/(T-residual) with weights (T-residual)^2.
    Conditional standard errors are obtained from the forest's per-tree
    prediction spread, and a doubly robust AIPW score supports the ATE and
    the calibration and AUTOC computations.

    This mirrors the orthogonal, doubly robust construction of EconML's
    CausalForestDML so that downstream code is engine-agnostic.
    """

    def __init__(self, n_trees: int, min_leaf: int, n_cv: int, seed: int):
        self.n_trees = n_trees
        self.min_leaf = min_leaf
        self.n_cv = n_cv
        self.seed = seed
        self.engine = "rlearner_fallback"

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "RLearnerCausalForest":
        n = len(X)
        self._mhat = np.zeros(n)
        self._ehat = np.zeros(n)
        kf = KFold(self.n_cv, shuffle=True, random_state=self.seed)
        for tr, te in kf.split(X):
            m = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=self.min_leaf,
                max_features="sqrt", random_state=self.seed, n_jobs=1,
            ).fit(X[tr], Y[tr])
            self._mhat[te] = m.predict(X[te])
            e = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=self.min_leaf,
                max_features="sqrt", random_state=self.seed + 1, n_jobs=1,
            ).fit(X[tr], T[tr])
            self._ehat[te] = e.predict_proba(X[te])[:, 1]
        self._ehat = np.clip(self._ehat, 0.05, 0.95)

        y_resid = Y - self._mhat
        t_resid = T - self._ehat
        eps = 1e-6
        pseudo = y_resid / np.where(np.abs(t_resid) < eps, eps, t_resid)
        weights = t_resid ** 2

        self._forest = RandomForestRegressor(
            n_estimators=self.n_trees, min_samples_leaf=self.min_leaf,
            max_features="sqrt", random_state=self.seed + 2, n_jobs=-1,
        )
        self._forest.fit(X, pseudo, sample_weight=weights)

        # Cross-fitted (out-of-fold) CATE predictions. Each fold's CATEs are
        # produced by a forest trained only on the other folds, so they are
        # honest (not evaluated on their own training data). These are used
        # for AUTOC and the calibration test, where in-sample CATEs would be
        # over-optimistic; the full-data forest above is used for the
        # reported per-individual CATEs and the best linear projection.
        self._cate_oof = np.zeros(len(X))
        kf2 = KFold(self.n_cv, shuffle=True, random_state=self.seed + 3)
        for tr, te in kf2.split(X):
            f = RandomForestRegressor(
                n_estimators=self.n_trees, min_samples_leaf=self.min_leaf,
                max_features="sqrt", random_state=self.seed + 4, n_jobs=-1,
            )
            f.fit(X[tr], pseudo[tr], sample_weight=weights[tr])
            self._cate_oof[te] = f.predict(X[te])

        # Store residuals and a doubly robust AIPW score for ATE / AUTOC.
        self._t_resid = t_resid
        self._y_resid = y_resid
        self._T = T
        self._Y = Y
        self._X = X
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        """Predicted CATE at the supplied covariate rows."""
        return self._forest.predict(X)

    def oof_cate(self) -> np.ndarray:
        """Cross-fitted (out-of-fold) CATE for the training sample.

        Honest predictions suitable for AUTOC and calibration. Falls back to
        the in-sample CATE only if cross-fitting was not performed.
        """
        return getattr(self, "_cate_oof", self.effect(self._X))

    def effect_se(self, X: np.ndarray) -> np.ndarray:
        """Per-row conditional SE from the per-tree prediction spread."""
        preds = np.stack(
            [tree.predict(X) for tree in self._forest.estimators_], axis=0
        )
        return preds.std(axis=0, ddof=1) / np.sqrt(self.n_trees)

    def aipw_scores(self) -> np.ndarray:
        """Doubly robust AIPW scores for the training sample.

        Uses the cross-fitted (out-of-fold) CATE so the scores and any
        AUTOC/calibration built on them are honest:
        score_i = tau_oof(x_i)
                  + (T_i - e_i)/(e_i(1-e_i)) * (Y_i - m_i - (T_i-e_i)*tau_oof(x_i))
        The mean of these scores is the doubly robust ATE.
        """
        tau = self.oof_cate()
        e = self._ehat
        weight = (self._T - e) / (e * (1 - e))
        residual = self._Y - self._mhat - (self._T - e) * tau
        return tau + weight * residual

    def ate(self) -> tuple[float, float]:
        """Doubly robust ATE and its standard error from the AIPW scores."""
        scores = self.aipw_scores()
        ate = float(np.mean(scores))
        se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        return ate, se


def fit_causal_forest(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int, logger: logging.Logger,
):
    """Fit a causal forest with EconML if available, else the R-learner.

    Returns an object exposing ``effect(X)`` and, for the fallback, the
    AIPW-score helpers. EconML's CausalForestDML exposes the same
    ``effect`` and an ``ate_`` and inference interface, which the callers
    use defensively.
    """
    cfg = CONFIG.causal_forest
    if ECONML_AVAILABLE:
        model_y = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=cfg.min_node_size,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        )
        model_t = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=cfg.min_node_size,
            max_features="sqrt", random_state=seed + 1, n_jobs=-1,
        )
        forest = CausalForestDML(
            model_y=model_y, model_t=model_t, discrete_treatment=True,
            n_estimators=cfg.n_trees, min_samples_leaf=cfg.min_node_size,
            max_samples=cfg.sample_fraction, honest=cfg.honest_splitting,
            cv=cfg.n_cv_folds, random_state=seed, n_jobs=-1,
        )
        forest.fit(Y, T, X=X)
        forest.engine = "econml_causalforestdml"  # type: ignore[attr-defined]
        return forest

    return RLearnerCausalForest(
        n_trees=cfg.n_trees, min_leaf=cfg.min_node_size,
        n_cv=cfg.n_cv_folds, seed=seed,
    ).fit(X, T, Y)


def estimate_ate(forest: Any, X: np.ndarray, logger: logging.Logger) -> dict[str, float]:
    """Estimate the ATE with a confidence interval, engine-agnostically."""
    if hasattr(forest, "ate") and callable(getattr(forest, "ate")):
        ate, se = forest.ate()  # R-learner fallback
    else:
        cate = forest.effect(X)
        ate = float(np.mean(cate))
        try:
            lo, hi = forest.ate_interval(X, alpha=0.05)
            se = float((hi - lo) / (2 * 1.96))
        except Exception:
            try:
                se = float(np.mean(forest.effect_inference(X).stderr ** 2) ** 0.5)
            except Exception:
                se = float("nan")
    ci_lo = ate - 1.96 * se if np.isfinite(se) else float("nan")
    ci_hi = ate + 1.96 * se if np.isfinite(se) else float("nan")
    p = (
        float(2 * scipy_stats.norm.sf(abs(ate / se)))
        if np.isfinite(se) and se > 0 else float("nan")
    )
    return {
        "ate": round(ate, 4),
        "se": round(se, 4) if np.isfinite(se) else float("nan"),
        "ci_lower": round(ci_lo, 4) if np.isfinite(ci_lo) else float("nan"),
        "ci_upper": round(ci_hi, 4) if np.isfinite(ci_hi) else float("nan"),
        "p_value": round(p, 4) if np.isfinite(p) else float("nan"),
        "significant": bool(np.isfinite(p) and (ci_lo > 0 or ci_hi < 0)),
    }


def cate_with_se(forest: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return CATE point estimates and per-row standard errors."""
    cate = forest.effect(X)
    if hasattr(forest, "effect_se"):
        se = forest.effect_se(X)
    else:
        try:
            se = forest.effect_inference(X).stderr
        except Exception:
            se = np.full_like(cate, np.nan)
    return np.asarray(cate, dtype=float), np.asarray(se, dtype=float)


# ===========================================================================
# CATE summary
# ===========================================================================
def summarize_cate(cate: np.ndarray, se: np.ndarray) -> dict[str, Any]:
    """Descriptive summary of the CATE distribution and its dispersion."""
    ci_lo = cate - 1.96 * se
    ci_hi = cate + 1.96 * se
    return {
        "n": int(len(cate)),
        "cate_mean": round(float(np.mean(cate)), 4),
        "cate_sd": round(float(np.std(cate, ddof=1)), 4),
        "cate_min": round(float(np.min(cate)), 4),
        "cate_q25": round(float(np.percentile(cate, 25)), 4),
        "cate_median": round(float(np.median(cate)), 4),
        "cate_q75": round(float(np.percentile(cate, 75)), 4),
        "cate_max": round(float(np.max(cate)), 4),
        "cate_iqr": round(float(np.percentile(cate, 75) - np.percentile(cate, 25)), 4),
        "pct_significant_positive": round(float(np.mean(ci_lo > 0)), 3),
        "pct_significant_negative": round(float(np.mean(ci_hi < 0)), 3),
    }


# ===========================================================================
# TOC / AUTOC (Yadlowsky et al. 2024)
# ===========================================================================
def compute_toc_autoc(
    cate: np.ndarray, scores: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    """Targeting Operator Characteristic curve and its area (AUTOC).

    Individuals are ranked by predicted CATE (descending). For each fraction
    q the TOC value is the mean doubly robust score among the top-q-ranked
    individuals minus the overall mean score, i.e. the excess treatment
    benefit obtained by prioritizing the highest-CATE individuals. AUTOC is
    the average TOC over q in (0, 1], a single-point summary of how well the
    CATE ranking concentrates treatment benefit (Yadlowsky et al., 2024).

    ``scores`` are the doubly robust (AIPW) per-individual scores whose mean
    is the ATE. Returns the TOC curve (a row per evaluated q) and the scalar
    AUTOC.
    """
    n = len(cate)
    order = np.argsort(-cate)
    ranked_scores = scores[order]
    overall = float(np.mean(scores))

    cumulative_mean = np.cumsum(ranked_scores) / np.arange(1, n + 1)
    toc = cumulative_mean - overall  # TOC(q) at q = i/n
    fractions = np.arange(1, n + 1) / n

    autoc = float(np.mean(toc))  # area under TOC over q in (0,1]

    # Report the curve on a coarse grid for the supplementary table.
    grid_q = np.linspace(0.05, 1.0, 20)
    grid_idx = np.clip((grid_q * n).astype(int) - 1, 0, n - 1)
    toc_df = pd.DataFrame({
        "fraction_treated": np.round(fractions[grid_idx], 3),
        "toc_value": np.round(toc[grid_idx], 4),
    })
    return toc_df, autoc


def bootstrap_autoc(
    cate: np.ndarray, scores: np.ndarray, n_bootstrap: int, seed: int,
) -> dict[str, float]:
    """Bootstrap SE and CLT-based test of H0: AUTOC = 0.

    Resamples individuals with replacement, recomputing AUTOC on each
    resample. The AUTOC satisfies a central-limit theorem (Yadlowsky et al.,
    2024), so the test statistic AUTOC / SE(AUTOC) is referred to the normal
    distribution. Returns the point estimate, bootstrap SE, CI, and p-value.
    """
    _, observed = compute_toc_autoc(cate, scores)
    n = len(cate)
    rng = np.random.default_rng(seed)

    def _iteration(s: int) -> float:
        local = np.random.default_rng(s)
        idx = local.integers(0, n, size=n)
        _, a = compute_toc_autoc(cate[idx], scores[idx])
        return a

    seeds = rng.integers(0, 1_000_000, size=n_bootstrap)
    boot = Parallel(
        n_jobs=CONFIG.hardware.n_cpu_workers,
        backend=CONFIG.hardware.parallel_backend,
    )(delayed(_iteration)(int(s)) for s in seeds)
    boot = np.array([b for b in boot if np.isfinite(b)])

    if len(boot) < n_bootstrap * 0.5:
        return {
            "autoc": round(observed, 4), "se": float("nan"),
            "ci_lower": float("nan"), "ci_upper": float("nan"),
            "t_statistic": float("nan"), "p_value": float("nan"),
            "heterogeneity_detected": False,
        }

    se = float(np.std(boot, ddof=1))
    t = observed / se if se > 0 else float("nan")
    p = float(2 * scipy_stats.norm.sf(abs(t))) if np.isfinite(t) else float("nan")
    return {
        "autoc": round(observed, 4),
        "se": round(se, 4),
        "ci_lower": round(observed - 1.96 * se, 4),
        "ci_upper": round(observed + 1.96 * se, 4),
        "t_statistic": round(t, 3) if np.isfinite(t) else float("nan"),
        "p_value": round(p, 4) if np.isfinite(p) else float("nan"),
        "heterogeneity_detected": bool(np.isfinite(p) and p < 0.05),
    }


# ===========================================================================
# Calibration test (Athey & Wager 2019)
# ===========================================================================
def calibration_test(
    cate: np.ndarray, scores: np.ndarray,
) -> dict[str, float]:
    """Athey-Wager best-linear-predictor calibration test.

    Regresses the doubly robust score on (i) the mean forest prediction
    (a column equal to the average CATE) and (ii) the centered forest
    prediction (the differential signal). The mean coefficient assesses
    whether the ATE is captured; the differential coefficient assesses
    whether the heterogeneity is well-calibrated (a value near 1 with a
    small p-value is evidence of genuine, correctly-scaled heterogeneity).
    """
    mean_pred = np.full_like(cate, float(np.mean(cate)))
    differential = cate - float(np.mean(cate))
    X = np.column_stack([mean_pred, differential])

    try:
        beta, _, rank, _ = np.linalg.lstsq(X, scores, rcond=None)
        resid = scores - X @ beta
        n = len(scores)
        dof = max(n - rank, 1)
        sigma2 = float(resid @ resid / dof)
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        b_mean, b_diff = float(beta[0]), float(beta[1])
        se_diff = float(se[1])
        t_diff = b_diff / se_diff if se_diff > 0 else float("nan")
        p_diff = (
            float(scipy_stats.norm.sf(t_diff))  # one-sided: H1 b_diff > 0
            if np.isfinite(t_diff) else float("nan")
        )
    except np.linalg.LinAlgError:
        return {"converged": False}

    return {
        "converged": True,
        "mean_prediction_coef": round(b_mean, 4),
        "differential_prediction_coef": round(b_diff, 4),
        "differential_se": round(se_diff, 4),
        "differential_p_value": round(p_diff, 4) if np.isfinite(p_diff) else float("nan"),
        "heterogeneity_detected": bool(np.isfinite(p_diff) and p_diff < 0.05),
    }


# ===========================================================================
# Best linear projection (Semenova & Chernozhukov 2021)
# ===========================================================================
def best_linear_projection(
    cate: np.ndarray, X: np.ndarray, covariate_names: list[str],
) -> pd.DataFrame:
    """Best linear projection of the CATE on the effect modifiers.

    Regresses the estimated CATE on the covariates to identify which
    baseline characteristics most strongly predict treatment-effect
    heterogeneity. Coefficients are reported with heteroskedasticity-robust
    (HC0) standard errors.
    """
    Xc = np.column_stack([np.ones(len(X)), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(Xc, cate, rcond=None)
        resid = cate - Xc @ beta
        # HC0 robust covariance.
        bread = np.linalg.pinv(Xc.T @ Xc)
        meat = Xc.T @ (Xc * (resid ** 2)[:, None])
        cov = bread @ meat @ bread
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        return pd.DataFrame()

    names = ["intercept"] + covariate_names
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        t = beta[i] / se[i] if se[i] > 0 else float("nan")
        p = float(2 * scipy_stats.norm.sf(abs(t))) if np.isfinite(t) else float("nan")
        rows.append({
            "covariate": name,
            "coefficient": round(float(beta[i]), 4),
            "se": round(float(se[i]), 4),
            "p_value": round(p, 4) if np.isfinite(p) else float("nan"),
            "significant": bool(np.isfinite(p) and p < 0.05),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Train-test split validation
# ===========================================================================
def train_test_validation(
    df: pd.DataFrame, covariates: list[str], logger: logging.Logger,
) -> dict[str, Any]:
    """Validate heterogeneity on a held-out test set, stratified by quartile.

    Fits the forest on a training split, predicts CATEs on the held-out
    test split, stratifies the test set into predicted-CATE quartiles, and
    reports the observed treated-minus-control difference within each
    quartile. A monotone increase across quartiles is evidence that the
    CATE ranking generalizes. Also reports the test-set AUTOC.
    """
    cfg = CONFIG.causal_forest
    seed = (
        CONFIG.reproducibility.root_seed
        + CONFIG.reproducibility.causal_forest_seed_offset + 99999
    )
    work = df[["treatment", "wellbeing"] + covariates].dropna().reset_index(drop=True)
    if len(work) < 200:
        logger.warning("Insufficient cases for train-test validation")
        return {"converged": False}

    idx = np.arange(len(work))
    train_idx, test_idx = train_test_split(
        idx, train_size=cfg.train_proportion, random_state=seed,
        stratify=work["treatment"],
    )
    Xtr = work.iloc[train_idx][covariates].to_numpy(float)
    Ttr = work.iloc[train_idx]["treatment"].to_numpy(int)
    Ytr = work.iloc[train_idx]["wellbeing"].to_numpy(float)

    try:
        forest = fit_causal_forest(Xtr, Ttr, Ytr, seed, logger)
    except Exception as exc:
        logger.warning("Train-test forest failed: %s", exc)
        return {"converged": False}

    Xte = work.iloc[test_idx][covariates].to_numpy(float)
    Tte = work.iloc[test_idx]["treatment"].to_numpy(int)
    Yte = work.iloc[test_idx]["wellbeing"].to_numpy(float)
    cate_te = forest.effect(Xte)

    quartiles = pd.qcut(cate_te, 4, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for q in range(int(np.nanmax(quartiles)) + 1):
        mask = quartiles == q
        if mask.sum() < 10:
            continue
        yq, tq = Yte[mask], Tte[mask]
        treated = float(np.mean(yq[tq == 1])) if np.any(tq == 1) else float("nan")
        control = float(np.mean(yq[tq == 0])) if np.any(tq == 0) else float("nan")
        diff = treated - control if np.isfinite(treated) and np.isfinite(control) else float("nan")
        rows.append({
            "quartile": q + 1,
            "n_in_quartile": int(mask.sum()),
            "predicted_cate_mean": round(float(np.mean(cate_te[mask])), 4),
            "observed_difference": round(diff, 4) if np.isfinite(diff) else float("nan"),
        })

    # Test-set AUTOC from a simple inverse-propensity score on the test split.
    e_test = float(np.mean(Tte))
    test_scores = Yte * (Tte - e_test) / (e_test * (1 - e_test))
    _, test_autoc = compute_toc_autoc(cate_te, test_scores)

    logger.info(
        "Train-test validation: n_train = %d, n_test = %d, test AUTOC = %.4f",
        len(train_idx), len(test_idx), test_autoc,
    )
    return {
        "converged": True,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_set_autoc": round(float(test_autoc), 4),
        "quartile_results": rows,
    }


# ===========================================================================
# Sensitivity: Rosenbaum bounds and the E-value
# ===========================================================================
def rosenbaum_bounds(
    ate_result: dict[str, float], logger: logging.Logger,
) -> pd.DataFrame:
    """Rosenbaum-style sensitivity of the ATE to a hidden confounder.

    For a range of gamma (the odds by which an unmeasured confounder could
    alter treatment assignment), the worst-case shift in the standardized
    effect is log(gamma) * SE; the upper-bound p-value at each gamma
    indicates how much hidden bias the finding can withstand before losing
    significance (Rosenbaum, 2002).
    """
    cfg = CONFIG.causal_forest
    ate, se = ate_result.get("ate"), ate_result.get("se")
    if ate is None or se is None or not np.isfinite(se) or se <= 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for gamma in cfg.rosenbaum_gamma_range:
        bias = np.log(gamma) if gamma > 0 else 0.0
        z_upper = (abs(ate) - bias * se) / se
        p_upper = float(scipy_stats.norm.sf(z_upper))
        rows.append({
            "gamma": gamma,
            "p_value_upper_bound": round(p_upper, 4),
            "robust_at_005": bool(p_upper < 0.05),
        })
    return pd.DataFrame(rows)


def e_value(ate_result: dict[str, float]) -> dict[str, Any]:
    """E-value for the ATE (VanderWeele & Ding, 2017).

    Converts the standardized effect to an approximate risk ratio
    (RR ~= exp(0.91 * d)) and reports the minimum strength of association,
    on the risk-ratio scale, that an unmeasured confounder would need with
    both treatment and outcome to explain away the point estimate, and to
    shift the confidence bound to the null.
    """
    ate = ate_result.get("ate")
    ci_lo, ci_hi = ate_result.get("ci_lower"), ate_result.get("ci_upper")
    if ate is None or not np.isfinite(ate):
        return {"converged": False}

    def _e(rr: float) -> float:
        if rr < 1:
            rr = 1 / rr
        return float(rr + np.sqrt(rr * (rr - 1)))

    rr_point = float(np.exp(0.91 * ate))
    e_point = _e(rr_point)

    if np.isfinite(ci_lo) and ci_lo > 0:
        e_ci = _e(float(np.exp(0.91 * ci_lo)))
    elif np.isfinite(ci_hi) and ci_hi < 0:
        e_ci = _e(float(np.exp(0.91 * ci_hi)))
    else:
        e_ci = 1.0  # CI crosses the null

    return {
        "converged": True,
        "ate": round(float(ate), 4),
        "rr_approximation": round(rr_point, 3),
        "e_value_point": round(e_point, 3),
        "e_value_ci_bound": round(e_ci, 3),
        "interpretation": (
            f"An unmeasured confounder would need a risk-ratio association "
            f"of at least {e_point:.2f} with both treatment and outcome to "
            f"explain away the point estimate."
        ),
    }


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
    """Execute the full causal-heterogeneity pipeline."""
    ensure_output_directories()
    set_global_seeds()
    logger = configure_logging()

    logger.info("Pipeline version: %s", CONFIG.pipeline_version)
    logger.info("Root seed: %d", CONFIG.reproducibility.root_seed)
    logger.info("Trees per forest: %d", CONFIG.causal_forest.n_trees)
    logger.info("Stage: 05_causal_heterogeneity")

    try:
        df = load_analysis_dataset(logger)
        for required in ("supervisor_support", "wellbeing"):
            if required not in df.columns:
                raise ValueError(f"Required variable {required} missing.")

        df = construct_treatment(df, logger)
        covariates = select_covariates(df, logger)

        work = df[["treatment", "wellbeing"] + covariates].dropna().reset_index(drop=True)
        X = work[covariates].to_numpy(float)
        T = work["treatment"].to_numpy(int)
        Y = work["wellbeing"].to_numpy(float)

        seed = (
            CONFIG.reproducibility.root_seed
            + CONFIG.reproducibility.causal_forest_seed_offset
        )

        # --- Phase 1: Fit the causal forest ---
        logger.info("=" * 72)
        logger.info("PHASE 1: Causal forest fit")
        logger.info("=" * 72)
        forest = fit_causal_forest(X, T, Y, seed, logger)
        engine = getattr(forest, "engine", "unknown")
        logger.info("Forest fitted (engine: %s)", engine)

        # --- Phase 2: ATE and CATE distribution ---
        logger.info("=" * 72)
        logger.info("PHASE 2: ATE and CATE distribution")
        logger.info("=" * 72)
        ate_result = estimate_ate(forest, X, logger)
        ate_result["engine"] = engine
        logger.info(
            "ATE = %.4f [%.4f, %.4f], p = %.4f",
            ate_result["ate"], ate_result["ci_lower"],
            ate_result["ci_upper"], ate_result["p_value"],
        )
        cate, cate_se = cate_with_se(forest, X)
        cate_summary = summarize_cate(cate, cate_se)
        cate_summary["engine"] = engine
        logger.info(
            "CATE: mean = %.4f, SD = %.4f, range [%.4f, %.4f], IQR = %.4f",
            cate_summary["cate_mean"], cate_summary["cate_sd"],
            cate_summary["cate_min"], cate_summary["cate_max"],
            cate_summary["cate_iqr"],
        )

        # Doubly robust scores for AUTOC and calibration.
        if hasattr(forest, "aipw_scores"):
            scores = forest.aipw_scores()
        else:
            e_hat = float(np.mean(T))
            scores = cate + (T - e_hat) / (e_hat * (1 - e_hat)) * (
                Y - Y.mean() - (T - e_hat) * cate
            )

        # Honest CATE for ranking in AUTOC and calibration: use cross-fitted
        # out-of-fold predictions when the engine provides them (the
        # fallback does), so the targeting evaluation is not optimistic.
        # In-sample CATEs are retained for the reported per-individual
        # estimates and the best linear projection.
        if hasattr(forest, "oof_cate"):
            cate_honest = forest.oof_cate()
        else:
            cate_honest = cate

        # --- Phase 3: TOC / AUTOC ---
        logger.info("=" * 72)
        logger.info("PHASE 3: TOC curve and AUTOC heterogeneity test")
        logger.info("=" * 72)
        toc_df, _ = compute_toc_autoc(cate_honest, scores)
        autoc_result = {"engine": engine}
        if CONFIG.causal_forest.compute_autoc:
            autoc_result.update(bootstrap_autoc(
                cate_honest, scores,
                n_bootstrap=CONFIG.causal_forest.autoc_n_bootstrap,
                seed=seed + 50000,
            ))
            logger.info(
                "AUTOC = %.4f (SE = %.4f), t = %.3f, p = %.4f, "
                "heterogeneity detected = %s (honest, cross-fitted CATE)",
                autoc_result.get("autoc", float("nan")),
                autoc_result.get("se", float("nan")),
                autoc_result.get("t_statistic", float("nan")),
                autoc_result.get("p_value", float("nan")),
                autoc_result.get("heterogeneity_detected", False),
            )

        # --- Phase 4: Calibration test ---
        logger.info("=" * 72)
        logger.info("PHASE 4: Athey-Wager calibration test")
        logger.info("=" * 72)
        calibration = calibration_test(cate_honest, scores)
        calibration["engine"] = engine
        if calibration.get("converged", False):
            logger.info(
                "Calibration: mean coef = %.3f, differential coef = %.3f "
                "(p = %.4f), heterogeneity = %s",
                calibration["mean_prediction_coef"],
                calibration["differential_prediction_coef"],
                calibration["differential_p_value"],
                calibration["heterogeneity_detected"],
            )

        # --- Phase 5: Best linear projection ---
        logger.info("=" * 72)
        logger.info("PHASE 5: Best linear projection of the CATE")
        logger.info("=" * 72)
        blp = best_linear_projection(cate, X, covariates)
        if len(blp) > 0:
            blp["engine"] = engine
            sig = blp[blp["significant"] & (blp["covariate"] != "intercept")]
            logger.info(
                "Best linear projection: %d covariate(s) significantly "
                "predict CATE heterogeneity: %s",
                len(sig), sig["covariate"].tolist(),
            )

        # --- Phase 6: Train-test validation ---
        logger.info("=" * 72)
        logger.info("PHASE 6: Train-test split validation")
        logger.info("=" * 72)
        validation = train_test_validation(df, covariates, logger)

        # --- Phase 7: Sensitivity analysis ---
        logger.info("=" * 72)
        logger.info("PHASE 7: Sensitivity (Rosenbaum bounds, E-value)")
        logger.info("=" * 72)
        rosenbaum = rosenbaum_bounds(ate_result, logger)
        evalue = e_value(ate_result)
        if evalue.get("converged", False):
            logger.info(
                "E-value (point) = %.2f, E-value (CI bound) = %.2f",
                evalue["e_value_point"], evalue["e_value_ci_bound"],
            )

        # --- Write tables ---
        logger.info("=" * 72)
        logger.info("Writing output tables")
        logger.info("=" * 72)
        write_table(pd.DataFrame([ate_result]), "table_s20_ate.csv", logger)
        write_table(pd.DataFrame([cate_summary]), "table_s21_cate_summary.csv", logger)
        write_table(toc_df, "table_s22_toc_curve.csv", logger)
        write_table(pd.DataFrame([autoc_result]), "table_s23_autoc.csv", logger)
        write_table(pd.DataFrame([calibration]), "table_s24_calibration.csv", logger)
        write_table(blp, "table_s25_best_linear_projection.csv", logger)
        if validation.get("converged", False) and validation["quartile_results"]:
            write_table(
                pd.DataFrame(validation["quartile_results"]),
                "table_s26_train_test_validation.csv", logger,
            )
        write_table(rosenbaum, "table_s27_rosenbaum.csv", logger)
        if evalue.get("converged", False):
            write_table(pd.DataFrame([evalue]), "table_s28_e_value.csv", logger)

        # Persist the per-individual CATE distribution for Stage 7.
        cate_df = pd.DataFrame({
            "case_index": work.index.to_numpy(),
            "treatment": T,
            "cate": np.round(cate, 4),
            "cate_se": np.round(cate_se, 4),
            "cate_ci_lower": np.round(cate - 1.96 * cate_se, 4),
            "cate_ci_upper": np.round(cate + 1.96 * cate_se, 4),
        })
        if "respid" in df.columns:
            cate_df["respid"] = df.loc[work.index, "respid"].to_numpy()
        cate_path = CONFIG.paths.models_dir / "cate_distribution.csv"
        cate_df.to_csv(cate_path, index=False)
        logger.info(
            "Per-individual CATE distribution written to %s (%d rows)",
            cate_path, len(cate_df),
        )

        # Selection metadata for downstream scripts.
        meta = {
            "engine": engine,
            "ate": ate_result["ate"],
            "ate_ci": [ate_result["ci_lower"], ate_result["ci_upper"]],
            "autoc": autoc_result.get("autoc", float("nan")),
            "autoc_p": autoc_result.get("p_value", float("nan")),
            "calibration_heterogeneity": calibration.get("heterogeneity_detected", False),
            "e_value_point": evalue.get("e_value_point", float("nan")),
            "cate_iqr": cate_summary["cate_iqr"],
        }
        meta_path = CONFIG.paths.models_dir / "causal_solution_meta.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        logger.info("Causal solution metadata written to %s", meta_path)

        # --- Final summary ---
        logger.info("=" * 72)
        logger.info("Causal heterogeneity analysis completed")
        logger.info("=" * 72)
        logger.info(
            "ATE = %.4f [%.4f, %.4f] | AUTOC = %.4f (p = %.4f) | "
            "calibration heterogeneity = %s | E-value = %.2f",
            ate_result["ate"], ate_result["ci_lower"], ate_result["ci_upper"],
            autoc_result.get("autoc", float("nan")),
            autoc_result.get("p_value", float("nan")),
            calibration.get("heterogeneity_detected", False),
            evalue.get("e_value_point", float("nan")),
        )
        return 0

    except Exception as exc:
        logger.exception("Causal heterogeneity analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
