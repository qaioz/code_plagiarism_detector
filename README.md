# Code Plagiarism Detection — RAG Systems

Homework 1 for Applied LLM Systems. Builds and evaluates four code-plagiarism detectors on a labeled dataset of paraphrased and unrelated Python snippets.

The full assignment spec is in `Assignment.md`. My personal notes / journey are in `HumanNotes.md`.

## What's in here

```
01_indexing.ipynb     # Fetch corpus, chunk, embed, build FAISS + BM25. Writes indexes/.
02_interactive.ipynb  # Loads indexes/. Exposes the four detectors as callable functions.
03_evaluation.ipynb   # Loads dataset + indexes/. Runs the eval, ablations, charts, discussion.

data/
  reference_corpus/   # 70 chunks across 5 GitHub repos (10 .py files each).
  test_dataset/       # 15 positives + 15 negatives, file-per-case with module docstrings.

indexes/              # Pre-built outputs of 01 (faiss, bm25, embeddings, chunks, metas).
results/              # Outputs of 03 (charts, metrics.json, predictions.json).
```

## Setup

1. **Python env** (used conda; any 3.11+ env is fine):
   ```bash
   pip install -r requirements.txt
   ```

2. **Gemini API key.** Copy the example and paste your key:
   ```bash
   cp .env.example .env
   # edit .env, set GEMINI_API_KEY=...
   ```
   The notebooks load `.env` via `python-dotenv`. No keys are hardcoded.

## How to run

The three notebooks are independent (each does its own imports + index loads), but the dependency direction is **01 → 02 / 03**.

- **Skip 01 unless rebuilding.** `indexes/` is already populated — 02 and 03 just read from it. 01 only needs to run if you change the corpus or chunking.
- **02** is for interactive testing. Run-All, then add a cell at the bottom and call `await detect_rag("...your snippet...")`. There's a smoke test at the end that runs all four detectors against a known positive and negative.
- **03** is the eval. Run-All produces every artifact in `results/` in ~50s with `MAX_CONCURRENCY = 20`.

To re-execute 03 from the command line:
```bash
jupyter nbconvert --to notebook --inplace --execute 03_evaluation.ipynb \
  --ExecutePreprocessor.timeout=2400
```

## What 03 produces

- `comparison_chart.png` — main 4-method × 4-metric bar chart
- `threshold_ablation.png` — embedding cosine-threshold sweep
- `k_ablation.png` — RAG / Hybrid `k` sweep
- `alpha_ablation.png` — Hybrid α (dense vs BM25) sweep
- `metrics.json` — every metric from every run
- `predictions.json` — per-case verdict from every detector

The discussion at the bottom of 03 reads against the actual measured numbers.

## Notes

- The assignment spec lists `data/test_dataset.json`. I went with a directory of `.py` files instead — easier to read, easier to edit individual cases, and 03's loader walks both folders. Functionally equivalent, arguably cleaner.
- `MAX_CONCURRENCY` in 03 is 20 (fits comfortably under tier-1 Gemini Flash's 1000 RPM cap). 01 and 02 stay at 6 because they don't need more.
