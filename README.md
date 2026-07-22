# Ranking Signal Analysis — FlyRank ML Capstone

> Which safe, page-level search signals are associated with SERP visibility (Top-10 placement)?

[![Notebook](https://img.shields.io/badge/notebook-Jupyter-F37626?logo=jupyter&logoColor=white)](work/capstone_ranking_signal_analysis.ipynb)
[![Python](https://img.shields.io/badge/python-3.9+-3776AB?logo=python&logoColor=white)](#reproducing)
[![License: Data Use](https://img.shields.io/badge/data-anonymized%20sample-lightgrey)](#data--scope-note)

This capstone was built as part of the **FlyRank ML Internship** (Ranking Signal Analysis lane). It takes a real, anonymized set of Google search results and asks a narrow, testable question: *of the signals we can safely observe on a page, which ones actually track with landing in the Top 10?*

The full write-up — motivation, methodology, results, and caveats — is published as a standalone research paper. The notebook here is the reproducible source of truth behind it.

**📄 Read the paper:** see [`submission/paper_url.txt`](submission/paper_url.txt) for the live deployed link.

---

## TL;DR — What the data actually shows

| Signal | Finding |
|---|---|
| **Recurring domain presence** | Strongest signal by far — Spearman ρ ≈ **‑0.54** vs. rank, point-biserial r ≈ **0.55** vs. Top‑10 (both p < .001) |
| **Wikipedia** | Drives much of that effect on its own — mean rank **3.9** vs. **14.6** for every other domain |
| **On-page title keyword matching** | **No measurable association** with rank or Top‑10 in this sample — a counter-intuitive result worth flagging, not hiding |
| **Snippet length** | Positively correlated univariately (r = 0.26), but the effect **shrinks toward zero once domain authority is controlled for** — a classic confound, not an independent signal |
| **Shorter titles** | Weakly associated with Top‑10 placement |
| **5-signal logistic regression** | **74.4% accuracy / 0.79 AUC** under leave-one-query-out validation, beating both a majority-class baseline (60.0%) and a single-signal heuristic (70.7%) |

All results are **observational and directional** — nothing here claims to reverse-engineer or prove how Google's live ranking algorithm works. See [Methodology](#methodology) for why, and [Limitations](#limitations--honest-caveats) for what this dataset can't tell you.

---

## Repo structure

```
work/
  capstone_ranking_signal_analysis.ipynb   main analysis: features -> correlations -> validated model -> findings
data/
  SEO_data_raw.csv                         raw SERP export (query, rank, title, h1, snippet, url, total_result)
  features.csv                             engineered signal table used by the notebook (375 rows x 24 columns)
paper/
  index.html                               the deployed research paper (static, GitHub-Pages-ready)
  assets/                                  charts referenced by the paper
submission/
  paper_url.txt                            single line: the exact deployed URL of the paper
```

---

## Dataset

- **375 rows** — 15 AI/ML-topic queries × top 25 ranked results each (e.g. *Machine learning*, *Clustering*, *Convolutional neural network*, *Reinforcement learning*, *Database*, *Chatbots*, …)
- **Target:** `top10` (binary — did this result land in positions 1–10?), 150 positive / 225 negative
- **Page-level, query-relative signals only** — no click logs, no private analytics, no ranking-API access
- Public-safe SERP export: query, rank, title, H1, snippet, URL, and query-level total-result count — no client names, private queries, or credentials

### Engineered features (`data/features.csv`)

| Category | Signals |
|---|---|
| Domain authority proxy | `domain_freq` (how often a domain recurs across the query set), `is_wikipedia` |
| Title matching | `title_query_coverage`, `title_exact_phrase`, `title_starts_with_query`, `title_len_chars`, `title_len_words` |
| Heading matching | `h1_query_coverage`, `h1_len_chars`, `h1_missing` |
| Snippet | `snippet_len_chars`, `snippet_printable_ratio`, `is_binary_snippet`, `is_pdf_result` |
| Outcome | `top10`, `top3`, `rank` |

**Data quality fix worth noting:** 13 of 375 raw rows had binary PDF/DOC bytes captured into the `snippet` field instead of extracted text (one cell alone was ~26M characters of raw stream data). Left uncleaned, this single outlier would have dominated any snippet-length signal and badly skewed standardization. These rows are detected via a printable-character-ratio check, their snippet length is treated as empty (not imputed, not dropped as a row), and they're separately flagged as PDF results.

---

## Methodology

1. **Load & engineer** — build the 24-column feature table from the raw SERP export (fully reproduced in the notebook, not just the pre-computed CSV).
2. **Signal report** — Spearman correlation of each signal against raw `rank`, and point-biserial correlation against the binary `top10` outcome, each with p-values.
3. **Baseline vs. model** — a 5-signal logistic regression (`domain_freq`, `title_len_words`, `snippet_len_chars`, `h1_query_coverage`, `title_starts_with_query`), validated with **Leave-One-Group-Out** grouped by query.
   - Why grouped, not a random split? Every query contributes 25 correlated rows (one per rank position). A random split would let the same query leak into both train and test, inflating apparent performance. LOGO guarantees every fold is tested on a query the model has never seen.
4. **Directional coefficients** — the same model refit on all 375 rows, reported only to rank which signals move the prediction most *within this dataset* — explicitly **not** presented as causal effects.

### Model performance (leave-one-query-out)

| Model | Accuracy | AUC |
|---|---|---|
| Baseline — always predict "not Top‑10" | 60.0% | — |
| Baseline — `domain_freq` median heuristic | 70.7% | — |
| **Logistic regression (5 signals)** | **74.4%** | **0.79** |

---

## Limitations & honest caveats

- **Sample size:** 15 queries × 25 ranks = 375 rows is small for a 15-fold grouped validation — treat accuracy/AUC as directional, not production-grade.
- **Single topic cluster:** all queries are AI/ML terms. The Wikipedia and domain-authority effects in particular are unlikely to generalize past this cluster without re-testing.
- **Correlation ≠ causation:** every finding here is an association observed in a static snapshot, not evidence of what Google's algorithm weighs or how it behaves over time.
- **Confounding is real and shown, not hidden:** the snippet-length flip between the univariate and multivariate results is reported explicitly because it's a useful lesson in why single-signal SEO claims are often confounded by authority.

---

## Reproducing

```bash
git clone https://github.com/Raunak24070/flyrank-machine-learning_ML_Capstone_Project_Ranking_Signal.git
cd flyrank-machine-learning_ML_Capstone_Project_Ranking_Signal

pip install pandas numpy scipy scikit-learn matplotlib jupyter

jupyter nbconvert --to notebook --execute \
  work/capstone_ranking_signal_analysis.ipynb \
  --output work/capstone_ranking_signal_analysis.ipynb
```

The notebook is self-contained: **load → engineer signals → correlate with rank/visibility → validate a model against baselines with a leakage-safe split → summarize findings.**

---

## Deploying the paper

1. `git add . && git commit -m "capstone: ranking signal analysis"`
2. Push to GitHub, then enable **GitHub Pages** (Settings → Pages → deploy from `main` / `/paper`, or move `paper/index.html` to `/docs/index.html`).
3. Put the resulting live URL — and only that URL — into [`submission/paper_url.txt`](submission/paper_url.txt).

---

## Tech stack

`pandas` · `numpy` · `scipy` (Spearman / point-biserial correlation) · `scikit-learn` (Logistic Regression, Leave-One-Group-Out, StandardScaler, Pipeline) · `matplotlib` · Jupyter

---

## Author

**Raunak** — AI & Data Science, YCCE Nagpur
Built for the FlyRank ML Internship capstone (Ranking Signal Analysis lane).
