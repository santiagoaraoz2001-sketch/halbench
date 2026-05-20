"""Corpus + calibration-endpoint loading.

Two sources, in priority order:
  1. Local path passed by user (--corpus path/to/items.parquet)
  2. HF Dataset (`santiagoaraoz/halbench-v2.1`, downloaded on first use)
  3. Package-bundled calibration endpoints (always available, no network needed)

The corpus itself is downloaded on demand because it's 5MB and rarely needed at
scoring time. The calibration endpoints (the production fingerprint) are
bundled in the package so scoring is fully offline once you have responses.
"""
from __future__ import annotations
import json
import os
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional


HF_DATASET = "santiagoaraoz/halbench-v2.1"


def load_endpoints() -> dict:
    """Load the locked calibration endpoints bundled inside the package.

    Returns the full endpoints document (schema_version, axis, endpoints, …).
    The 'endpoints' key is a {cell_field: {defer/soft/hard: {mean_raw_M5, ...}}} map.
    """
    with resources.files("halbench.data").joinpath("calibration_endpoints.json").open("r") as f:
        return json.load(f)


def load_corpus(source: Optional[str] = None) -> list[dict]:
    """Load the 3,600-item HalBench corpus.

    Args:
        source: Optional path to a local items.parquet, items.jsonl, or a
                directory containing items/*.json. If None, downloads from
                the HF Dataset (`santiagoaraoz/halbench-v2.1`).

    Returns:
        List of item dicts, each with at least: item_id, cell, field,
        cell_field, prompt.
    """
    if source:
        return _load_corpus_local(source)
    return _load_corpus_hf()


def _load_corpus_local(source: str) -> list[dict]:
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(source)
    if p.is_dir():
        # Treat as items/ directory of JSON files
        items_dir = p / "items" if (p / "items").is_dir() else p
        out = []
        for fn in sorted(os.listdir(items_dir)):
            if fn.endswith(".json"):
                out.append(json.load(open(items_dir / fn)))
        return out
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in open(p) if line.strip()]
    if p.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas needed to load .parquet (pip install pandas)")
        return pd.read_parquet(p).to_dict("records")
    raise ValueError(f"Unsupported corpus source: {p}")


def _load_corpus_hf() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "datasets package needed to download corpus from HF "
            "(pip install 'halbench[hf]' or pip install datasets). "
            "Alternatively, download items.parquet manually and pass it via `source=`."
        )
    ds = load_dataset(HF_DATASET, data_files="items.parquet", split="train")
    return list(ds)


def iter_prompts(items: Iterable[dict]) -> Iterable[tuple[str, str]]:
    """Yield (item_id, prompt_text) pairs in deterministic order."""
    for it in items:
        yield it["item_id"], it["prompt"]
