# Deploying HalBench V2.1 to Hugging Face

This doc covers deploying both the **Space** (interactive Gradio app) and the **Dataset** (parquet files) under the `specific-labs` organization (or your personal account). Both are set up to launch **private** for iteration, then flipped public.

---

## Prerequisites

```bash
pip install --upgrade huggingface_hub
huggingface-cli login   # paste your HF write-access token
```

You'll need:
- A write-access token from https://huggingface.co/settings/tokens
- A target namespace (your username, e.g. `santiagoaraoz`, or an org you can write to, e.g. `specific-labs`). Replace `<NS>` below with your choice.

---

## A. Deploying the Space (Gradio app)

The Space lives at `benchmark/space/`. To deploy:

### A1. Create the Space (one-time)

```bash
# From the project root:
cd "/Users/santiagoaraoz/Desktop/Specific Labs/P6_Hallucination/Benchmarks/halbench_v2"

huggingface-cli repo create halbench-v2.1 \
  --type space \
  --space_sdk gradio \
  --private \
  --organization <NS>    # omit --organization to deploy under your username
```

This creates `https://huggingface.co/spaces/Specific-Labs/halbench` as a **private** Space.

### A2. Push the app + data (every update)

```bash
cd benchmark/space

# Initialize as git repo on first push
git init
git remote add origin https://huggingface.co/spaces/Specific-Labs/halbench
git lfs install
git lfs track "data/*.jsonl"     # responses_scores.jsonl is 37MB
git lfs track "data/*.json"

# Stage everything in the space directory
git add app.py requirements.txt README.md .gitattributes
git add data/

# Commit + push
git commit -m "HalBench V2.1 — initial Space deployment"
git push -u origin main
```

The Space will rebuild automatically (takes ~3 min on free CPU tier). Watch the build log at:
`https://huggingface.co/spaces/Specific-Labs/halbench?logs=build`

### A3. Flip to public when ready

In the Space UI: **Settings → Visibility → Public**. Or via API:

```python
from huggingface_hub import HfApi
HfApi().update_repo_visibility(repo_id="<NS>/halbench-v2.1", repo_type="space", private=False)
```

---

## B. Deploying the Dataset

The dataset lives at `benchmark/dataset/build/` (built by `build_dataset.py`).

### B1. Create the Dataset repo

```bash
huggingface-cli repo create halbench-v2.1 \
  --type dataset \
  --private \
  --organization <NS>
```

### B2. Upload parquet files

```bash
cd benchmark/dataset/build

git init
git remote add origin https://huggingface.co/datasets/Specific-Labs/halbench
git lfs install
git lfs track "*.parquet"

git add README.md meta.json .gitattributes
git add *.parquet
git commit -m "HalBench V2.1 — initial dataset upload"
git push -u origin main
```

### B3. Verify load

```python
from datasets import load_dataset
ds = load_dataset("<NS>/halbench-v2.1", token="hf_...")  # token needed while private
print(ds["items"][0])
```

### B4. Wire the Space to the Dataset

Once both are uploaded, update `benchmark/space/app.py` so it loads from the HF Dataset instead of bundled `data/`. Change the data loading block to:

```python
from datasets import load_dataset
DS = load_dataset("<NS>/halbench-v2.1")
ITEMS = list(DS["items"])
RESP_SCORES = list(DS["responses_scores"])
# ... etc
```

This is **optional** — the bundled approach (current setup) works fine and is simpler. The dataset link is mainly for external researchers who want to download the data programmatically without going through the Space.

---

## Updating either repo later

Both deploy as standard git repos. To update:

```bash
# Rebuild local data
python3 benchmark/space/prepare_space_data.py
python3 benchmark/dataset/build_dataset.py

# Push Space updates
cd benchmark/space
git add app.py data/
git commit -m "Add 2 new models to leaderboard"
git push

# Push Dataset updates
cd ../dataset/build
git add *.parquet README.md
git commit -m "v2.1.1 — add 2 new models"
git push
```

The Space auto-rebuilds on push. The Dataset reflects the new files immediately.

---

## Troubleshooting

**"Repository not found" on push** → You aren't logged in to the right namespace. Run `huggingface-cli whoami` to check; re-run `huggingface-cli login` if needed.

**Space build fails with "no app.py"** → Make sure you're pushing from inside `benchmark/space/` (not from the project root). The Space repo's root must contain `app.py` and `README.md` at the top level.

**Space build OOMs** → Free CPU tier has 16GB. Current data bundle is 47MB, well under. If you bundle response embeddings later you may need to upgrade to a paid tier or chunk-load on demand.

**Plotly figures don't render** → Make sure `gradio>=5.0` is in `requirements.txt`. Older Gradio versions render plotly via static HTML which loses interactivity.

**Item Explorer dropdown is slow** → The dropdown caps at 1,000 items for UI responsiveness. If you need full search, swap `gr.Dropdown` for `gr.Textbox` with an autocomplete callback.

---

## Cost estimate

- **Space (free CPU tier)**: $0/month for current scope. No model inference happens server-side — all data is precomputed.
- **Dataset**: $0/month. HF hosts parquet files free under 50GB.
- **If you add inference** (e.g., "submit a model and we score it live"): Each new model is ~$0.50 of HF Inference (3,600 Harrier embeddings @ batch=4). Trivial.

Total recurring cost: **$0** for the current scope.
