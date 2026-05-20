# HalBench V2.1 — Leaderboard Submissions

Each file in `submissions/` is a complete scoring of one model against the
3,600-item corpus. They drive the public leaderboard at the HF Space.

## How to add your model

1. **Install the package**: `pip install halbench` (or `pip install -e .` from this repo)
2. **Set credentials**:
   - `OPENROUTER_API_KEY` (if using the openrouter backend)
   - `HF_TOKEN` (always required — for Harrier embeddings)
3. **Run the full corpus** against your model:
   ```bash
   halbench run \
     --model your-org/your-model \
     --backend openrouter \
     --out ./my_run
   ```
   This produces `my_run/responses.jsonl` (the chat outputs) and
   `my_run/scores.jsonl` (the graded scores). Cost depends on the model —
   roughly $5–$50 of inference for most frontier APIs.
4. **Verify locally**:
   ```bash
   halbench verify my_run/scores.jsonl
   ```
   This re-runs scoring and confirms reproducibility within ±0.005.
5. **Submit a PR**:
   - Fork the repo
   - Copy `my_run/scores.jsonl` to `leaderboard/submissions/<your_model_slug>.jsonl`
   - The slug convention is `provider__model-name` (e.g. `anthropic__claude-sonnet-4.6`)
   - Open a PR. CI verifies your submission; on green review it gets merged
     and the Space rebuilds with your model included.

## What the CI checks

When you open a PR that touches `leaderboard/submissions/`, the
`verify-submission` GitHub Action will:

1. Install this package fresh
2. Re-score every response in your submission using the locked Harrier embedder
   + bundled calibration endpoints
3. Confirm submitted scores match the recomputed scores within ±0.005
4. Confirm the submission covers all 3,600 corpus items (no skipping)
5. Block merge if any check fails, with a comment explaining why

You cannot game the leaderboard by hand-editing scores — every number is
verifiable from the corresponding `response_text`.

## What's currently here

| Submission | Mean | Notes |
|---|---:|---|
| anthropic__claude-sonnet-4.6.jsonl | 0.716 | 17 silent-refusal anomalies on B2_SK |
| x-ai__grok-4.3.jsonl | 0.564 | |
| google__gemini-3.1-pro-preview.jsonl | 0.362 | |
| openai__gpt-5.4.jsonl | 0.348 | |
