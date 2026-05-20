"""End-to-end run: corpus → backend → responses.jsonl → scores.jsonl."""
from __future__ import annotations
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from halbench.backends import get_backend
from halbench.corpus import load_corpus
from halbench.scoring import score_responses


def run_model(
    backend_name: str,
    model: str,
    out_dir: str,
    corpus_source: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system: Optional[str] = None,
    concurrency: int = 8,
    limit: Optional[int] = None,
    resume: bool = True,
    verbose: bool = True,
    score: bool = True,
    hf_token: Optional[str] = None,
) -> dict:
    """Run a model against the full HalBench corpus.

    Args:
        backend_name: e.g. "openrouter"
        model: model id as the backend expects, e.g. "anthropic/claude-sonnet-4.6"
        out_dir: directory to write responses.jsonl + (optional) scores.jsonl
        corpus_source: optional local corpus path; else downloads from HF Dataset
        temperature, max_tokens, system: model sampling params
        concurrency: parallel chat completions
        limit: only run first N items (for smoke tests)
        resume: skip items already present in responses.jsonl (default True)
        score: also score the responses (calls halbench score under the hood)
        hf_token: needed for scoring (Harrier embedder)

    Returns:
        Summary dict with counts + cost.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    resp_path = out / "responses.jsonl"
    score_path = out / "scores.jsonl"

    backend = get_backend(backend_name, model=model)
    items = load_corpus(corpus_source)
    if limit:
        items = items[:limit]

    # Resume support
    done_ids: set[str] = set()
    if resume and resp_path.exists():
        for line in open(resp_path):
            try:
                r = json.loads(line)
                if r.get("response_text") and not r.get("error"):
                    done_ids.add(r["item_id"])
            except Exception:
                pass
        if verbose:
            print(f"  Resume: {len(done_ids)} already done, {len(items) - len(done_ids)} remaining")

    todo = [it for it in items if it["item_id"] not in done_ids]
    if verbose:
        print(f"Running {len(todo)} items × {model} via {backend_name}  (concurrency={concurrency})")

    t0 = time.time()
    n_ok = 0
    n_err = 0
    total_cost = 0.0

    # Append-only — failure halfway is recoverable via --resume
    with open(resp_path, "a") as f, ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {
            ex.submit(backend.chat, it["prompt"], temperature=temperature,
                       max_tokens=max_tokens, system=system): it
            for it in todo
        }
        for i, fut in enumerate(as_completed(futs), 1):
            it = futs[fut]
            try:
                resp = fut.result()
                row = {
                    "item_id": it["item_id"],
                    "cell_field": it["cell_field"],
                    "model": model,
                    "backend": backend_name,
                    "response_text": resp.response_text,
                    "finish_reason": resp.finish_reason,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "usage_in_tokens": resp.usage_in_tokens,
                    "usage_out_tokens": resp.usage_out_tokens,
                    "cost_usd": resp.cost_usd,
                    "latency_s": resp.latency_s,
                    "timestamp": time.time(),
                }
                if resp.error:
                    row["error"] = resp.error
                    n_err += 1
                else:
                    n_ok += 1
                total_cost += resp.cost_usd
                f.write(json.dumps(row) + "\n")
                f.flush()
                if verbose and i % 50 == 0:
                    elapsed = time.time() - t0
                    rate = i / max(elapsed, 0.01)
                    eta = (len(todo) - i) / max(rate, 0.01)
                    print(f"  [{i}/{len(todo)}] ok={n_ok} err={n_err} "
                          f"cost=${total_cost:.2f} rate={rate:.1f}/s eta={eta:.0f}s")
            except Exception as e:
                n_err += 1
                f.write(json.dumps({"item_id": it["item_id"], "error": str(e)[:300]}) + "\n")
                f.flush()

    if verbose:
        print(f"\n  Done: {n_ok} OK, {n_err} errors, ${total_cost:.2f}, "
              f"{time.time()-t0:.0f}s")
        print(f"  Wrote {resp_path}")

    summary: dict = {
        "model": model, "backend": backend_name,
        "n_ok": n_ok, "n_err": n_err,
        "total_cost_usd": total_cost,
        "responses_path": str(resp_path),
        "elapsed_s": time.time() - t0,
    }

    if score:
        if verbose:
            print(f"\nScoring responses...")
        score_responses(str(resp_path), out_jsonl=str(score_path),
                         hf_token=hf_token, verbose=verbose)
        summary["scores_path"] = str(score_path)

    return summary
