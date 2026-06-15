# Individual Differences in Doctoral Learning Adaptation and Well-Being

Reproducible analysis pipeline for a mixed-methods study of academic pressure,
supervisor support, career uncertainty, and well-being among Chinese PhD
students, using the 2025 Nature Graduate Survey. The pipeline combines latent
profile analysis, a causal-forest estimate of the effect of supervisor support
on well-being (with honest, out-of-fold heterogeneity evaluation), a
generative-AI moderation analysis, and a joint-display integration.

## What this code does

Seven stages run in dependency order. Each is a standalone script with its own
`main()` and logging; `main.py` orchestrates them as isolated subprocesses.

| Stage | Script | Purpose | Key outputs |
|------:|--------|---------|-------------|
| 1 | `01_data_preparation.py` | Build the analysis dataset and codebook from the raw survey | `data/chinese_phd_dataset.csv`, `data/codebook.json` |
| 2 | `02_psychometric_validation.py` | Reliability, CFA, invariance (reflective scales); VIF/endorsement (formative indices) | `outputs/tables/table_s2_reliability.csv` |
| 3 | `03_latent_profile_analysis.py` | Gaussian-mixture latent profiles (K = 1–6), enumeration, selection | `outputs/models/profile_assignments.csv` |
| 4 | `04_predictive_modeling.py` | Generative-AI moderation: 9 interaction tests + gradient boosting | `outputs/tables/table_s14_classical_moderation.csv` |
| 5 | `05_causal_heterogeneity.py` | Causal forest, ATE, CATEs, AUTOC, calibration, Rosenbaum/E-value | `outputs/models/cate_distribution.csv` |
| 6 | `06_qualitative_nlp.py` | Categorical cross-tabs against profiles (neural topic-model path retained but dormant) | `outputs/tables/table_s29_text_availability.csv` |
| 7 | `07_mixed_methods_integration.py` | Joint display + profile-level CATE summary | `outputs/tables/table_s36_joint_display.csv` |

**Dependency graph:** 1 → {2, 3, 4, 5}; 6 needs {1, 3}; 7 needs {3, 5, 6}.
`main.py` checks each stage's declared inputs before running it and aborts early
with a clear error if an upstream output is missing.

## Requirements

- Python 3.10 or later
- Dependencies in `requirements.txt`

`semopy` and `statsmodels` (Stage 2) and the `bertopic` / `sentence-transformers`
/ `umap-learn` / `hdbscan` group (Stage 6) are heavier and partly optional — see
the comments in `requirements.txt`. Stage 2 falls back to a built-in
maximum-likelihood single-factor CFA and a NumPy VIF when `semopy`/`statsmodels`
are absent, but the published numbers assume they are installed. The neural
topic-modeling group is **not invoked** on the anonymized public data (no open
text survives anonymization); it is retained for a future text-bearing wave.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

This repository contains **code only**. The analysis uses a publicly available,
fully anonymized dataset, the **2025 Nature Graduate Survey**, archived on
figshare:

> https://doi.org/10.6084/m9.figshare.30084739

To reproduce the results:

1. Download the raw survey file from the figshare record above.
2. Place it at `data/raw/Nature Graduate Survey_Raw_Data_anonymised for publishing.xlsx`
   (the pipeline reads the `Codes` worksheet).
3. Run Stage 1, which writes the derived `data/chinese_phd_dataset.csv` and
   `data/codebook.json` that every later stage consumes.

The `data/` directory is git-ignored by default; do not commit the raw survey or
any derived datasets.

## Expected layout

```
.
├── 01_data_preparation.py
├── 02_psychometric_validation.py
├── 03_latent_profile_analysis.py
├── 04_predictive_modeling.py
├── 05_causal_heterogeneity.py
├── 06_qualitative_nlp.py
├── 07_mixed_methods_integration.py
├── configs.py            # single source of truth for paths, seeds, item sets
├── main.py               # orchestrator
├── requirements.txt
├── data/                 # you create this; not committed
│   └── raw/              # place the figshare .xlsx here
└── outputs/              # created on first run (tables, models, logs)
```

`configs.py` resolves all paths relative to the project root and is the single
place to change inputs, output locations, seeds, and item definitions.

## Usage

Run the whole pipeline:

```bash
python main.py
```

Useful flags (from `main.py`):

```bash
python main.py --list               # list the stages and exit
python main.py --dry-run            # show the plan and check inputs; run nothing
python main.py --stages 1 3 5       # run a subset (dependencies still checked)
python main.py --from-stage 3       # run stage 3 onward
python main.py --continue-on-error  # keep going after a stage fails
```

Stages can also be run individually, in dependency order:

```bash
python 01_data_preparation.py
python 03_latent_profile_analysis.py
python 05_causal_heterogeneity.py
# ...
```

## Reproducibility

A single fixed root seed governs every stochastic procedure, with
stage-specific offsets so independent stages draw distinct but reproducible
random streams (see `configs.py`). Each stage records the estimator and library
versions it used. Given the same input dataset and the pinned dependency
versions, every reported quantity regenerates deterministically.

## Citation

If you use this code, please cite the associated article. *(Add the full
citation / DOI here once the paper is published.)*

## License

MIT — see [LICENSE](LICENSE).
