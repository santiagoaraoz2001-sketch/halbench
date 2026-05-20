# Deploying the halbench package + repo to GitHub

This walks through pushing the package to `github.com/santiagoaraoz/halbench` and (optionally) publishing it to PyPI.

## Prerequisites

```bash
# GitHub CLI (or use https://github.com/new in the browser)
brew install gh  # if not already installed
gh auth login    # follow prompts; pick HTTPS + browser auth

# Verify
gh auth status
```

## 1. Create the GitHub repo (one-time)

```bash
cd "/Users/santiagoaraoz/Desktop/Specific Labs/P6_Hallucination/Benchmarks/halbench_v2/benchmark/halbench-pkg"

# Init local git
git init
git add .
git commit -m "Initial commit: HalBench V2.1 runner + leaderboard"

# Create the GitHub repo and push
gh repo create halbench \
  --public \
  --source . \
  --description "Sycophancy benchmark for frontier LLMs — runner, scoring, and CI-verified leaderboard submissions" \
  --homepage "https://huggingface.co/spaces/Specific-Labs/halbench" \
  --push
```

After this, the repo lives at `https://github.com/santiagoaraoz/halbench`.

## 2. Configure CI secrets

The `verify-submission.yml` workflow needs `HF_TOKEN` to call the Harrier
embedder during PR verification. Add it as a repo secret:

```bash
gh secret set HF_TOKEN  # paste your HF token when prompted
```

Or via the web UI: **Settings → Secrets and variables → Actions → New repository secret**.

## 3. (Optional) Publish to PyPI

If you want `pip install halbench` to work for everyone:

```bash
# Install build tools
python3 -m pip install --upgrade build twine

# Build wheel + sdist
python3 -m build

# Upload to TestPyPI first (sanity check)
twine upload --repository testpypi dist/*
# Try it:  pip install --index-url https://test.pypi.org/simple/ halbench

# Then upload to real PyPI
twine upload dist/*
```

You'll need a PyPI account and an API token from https://pypi.org/manage/account/token/.

If you don't publish to PyPI, users can still install via:

```bash
pip install git+https://github.com/santiagoaraoz/halbench.git
```

## 4. Smoke-test the deployed flow

After the repo is live, anyone can:

```bash
# Install
pip install git+https://github.com/santiagoaraoz/halbench.git
# (or pip install halbench if you published)

# Set credentials
export OPENROUTER_API_KEY=sk-or-...
export HF_TOKEN=hf_...

# Run a 50-item smoke test
halbench run --model anthropic/claude-sonnet-4.6 --limit 50 --out ./test
# → ./test/responses.jsonl + ./test/scores.jsonl

# Verify it's reproducible
halbench verify ./test/scores.jsonl --expected-n 50
```

## 5. Link up the Space + Dataset + GitHub repo

Once all three are live, edit each to cross-link:

- **Space README** — already links to the dataset & github (see `benchmark/space/README.md`)
- **Dataset README** — already links to the space (see `benchmark/dataset/build/README.md`)
- **GitHub README** — already links to both (see this repo's `README.md`)

If your final usernames differ from the placeholder `santiagoaraoz`, run a find/replace
across all three READMEs.

## 6. Updating the leaderboard

When new submissions come in via PRs (and pass CI), merge them. To rebuild the
Space's data bundle with the new model:

```bash
# Back in the halbench_v2 repo
python3 benchmark/space/prepare_space_data.py    # rebuilds data/ from leaderboard/submissions/
python3 benchmark/dataset/build_dataset.py       # rebuilds parquet bundle

# Push to HF (see benchmark/space/DEPLOY.md)
cd benchmark/space
git add data/
git commit -m "Add submission: foo/bar-12b"
git push   # Space auto-rebuilds
```

The submission flow can be automated end-to-end later (PR merge → Space rebuild
via GitHub Actions → push to HF) but for now it's manual.

## Troubleshooting

**`gh repo create` fails: "name already exists"** → A repo with that name is
already on your account or the org. Pick a different name or delete the old
one: `gh repo delete santiagoaraoz/halbench`.

**CI verify-submission fails on a real submission** → Most common cause is the
submitter used a different embedder. Run `halbench verify` locally with `--quiet`
disabled to see the per-item drift.

**`pip install -e .` fails on Python 3.9** → We require Python 3.10+. Bump
their Python version or relax the requirement in `pyproject.toml` if you want
3.9 support (the code itself is compatible; we use 3.10+ syntax in a few type
hints that are easy to downgrade).
