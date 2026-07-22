# Ranking Signal Analysis — FlyRank ML Capstone

**Lane:** Ranking Signal Analysis
**Question:** Which safe, page-level search signals are associated with SERP visibility (Top-10 placement)?

## Deployed paper
See `submission/paper_url.txt` for the live URL. (Replace the placeholder with your GitHub Pages / hosting URL after deploying `paper/index.html`.)

## Repo structure
```
work/                                    every assignment + capstone notebook
  capstone_ranking_signal_analysis.ipynb main analysis: features -> correlations -> validated model -> findings
data/
  SEO_data_raw.csv                       raw SERP export (query, rank, title, h1, snippet, url, total_result)
  features.csv                           engineered signal table used by the notebook
paper/
  index.html                             the deployed research paper (static, GitHub-Pages-ready)
submission/
  paper_url.txt                          single line: the exact deployed URL of the paper (required)
```

## Reproducing
```
pip install pandas numpy scipy scikit-learn matplotlib jupyter
jupyter nbconvert --to notebook --execute work/capstone_ranking_signal_analysis.ipynb --output work/capstone_ranking_signal_analysis.ipynb
```

## Data & scope note
Raw export is a public-safe SERP sample (query, rank 1-25, title/H1/snippet, URL, query-level total-result count) — no client names, private queries, or credentials. All analysis is observational/directional; nothing here claims to reveal or prove Google's ranking algorithm.

## Deploying the paper
1. `git init && git add . && git commit -m "capstone: ranking signal analysis"`
2. Push to GitHub, enable **GitHub Pages** on the repo (Settings → Pages → deploy from `main` / `/paper` or `/docs`, or move `paper/index.html` to `/docs/index.html`).
3. Put the resulting URL — and only that URL — in `submission/paper_url.txt`.
