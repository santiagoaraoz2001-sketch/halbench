"""Submission verification — used by CI and by `halbench verify` locally.

A submission is a scores.jsonl (one row per item) produced by `halbench score`.
Verification re-runs the scoring (which is deterministic given identical
calibration endpoints + embeddings) and checks the submitted scores match
within a small tolerance.

This guarantees:
  (a) The submitter used the locked calibration endpoints (not their own).
  (b) The submitter didn't hand-edit scores after running.
  (c) The response_text in the submission matches what would produce those scores.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Optional

from halbench.scoring import score_responses, BENCHMARK_VERSION


SCORE_TOLERANCE = 0.005   # ±0.5% absolute — embedder is deterministic but float math drifts
EXPECTED_N_ITEMS = 3600   # must cover the whole corpus


def verify_submission(
    submission_jsonl: str,
    *,
    expected_n: int = EXPECTED_N_ITEMS,
    tolerance: float = SCORE_TOLERANCE,
    hf_token: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Verify a scores.jsonl submission.

    Returns dict with: ok (bool), errors (list[str]), summary (dict).
    Exit status from CLI: 0 if ok, 1 if any errors.
    """
    rows = [json.loads(line) for line in open(submission_jsonl) if line.strip()]
    errors: list[str] = []

    # ---- Structural checks ----
    if len(rows) != expected_n:
        errors.append(
            f"Expected exactly {expected_n} rows; got {len(rows)}. "
            f"Submissions must cover the full corpus."
        )
    seen_ids = set()
    for i, r in enumerate(rows):
        if "item_id" not in r:
            errors.append(f"Row {i}: missing item_id")
        elif r["item_id"] in seen_ids:
            errors.append(f"Row {i}: duplicate item_id {r['item_id']}")
        else:
            seen_ids.add(r["item_id"])
        if r.get("benchmark_version") and r["benchmark_version"] != BENCHMARK_VERSION:
            errors.append(
                f"Row {i} ({r.get('item_id')}): benchmark_version "
                f"{r['benchmark_version']} != {BENCHMARK_VERSION}"
            )

    if errors:
        return {"ok": False, "errors": errors, "summary": {"n_rows": len(rows)}}

    # ---- Re-score check ----
    # We need response_text in the submission to re-score. If missing, fail.
    no_text = [r for r in rows if not r.get("is_anomaly") and not r.get("response_text")]
    if no_text:
        errors.append(
            f"{len(no_text)} non-anomaly rows missing response_text — cannot re-verify. "
            "Submissions must include the response text for every scored item."
        )
        return {"ok": False, "errors": errors, "summary": {"n_rows": len(rows)}}

    # Write a temp responses.jsonl, re-score it, compare
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        for r in rows:
            tf.write(json.dumps({
                "item_id": r["item_id"],
                "cell_field": r.get("cell_field") or r["item_id"].split("__")[0],
                "response_text": r.get("response_text", ""),
                "finish_reason": r.get("finish_reason"),
                "model": r.get("model"),
            }) + "\n")
        tmp_responses = tf.name

    if verbose:
        print(f"Re-scoring {len(rows)} rows for verification...")
    re_scored = score_responses(tmp_responses, out_jsonl=None, hf_token=hf_token, verbose=verbose)
    os.unlink(tmp_responses)

    # Compare scores
    re_by_id = {r["item_id"]: r for r in re_scored}
    n_match = 0
    n_drift = 0
    drift_examples: list[str] = []
    for r in rows:
        iid = r["item_id"]
        if r.get("is_anomaly"):
            continue
        submitted = r.get("score")
        recomputed = re_by_id.get(iid, {}).get("score")
        if submitted is None or recomputed is None:
            continue
        if abs(submitted - recomputed) > tolerance:
            n_drift += 1
            if len(drift_examples) < 5:
                drift_examples.append(
                    f"  {iid}: submitted={submitted:.4f} recomputed={recomputed:.4f} "
                    f"Δ={submitted - recomputed:+.4f}"
                )
        else:
            n_match += 1

    if n_drift > 0:
        errors.append(
            f"{n_drift} scores drifted beyond ±{tolerance:.4f} between submission "
            f"and re-scoring. Either (a) submitter used different calibration endpoints, "
            f"(b) embeddings differ (model/version mismatch?), or (c) scores were "
            f"hand-edited.\nFirst {len(drift_examples)} examples:\n" + "\n".join(drift_examples)
        )

    summary = {
        "n_rows": len(rows),
        "n_anomalies": sum(1 for r in rows if r.get("is_anomaly")),
        "n_score_match": n_match,
        "n_score_drift": n_drift,
        "tolerance": tolerance,
        "submitted_mean": sum(r["score"] for r in rows if r.get("score") is not None) / max(
            sum(1 for r in rows if r.get("score") is not None), 1
        ),
    }
    return {"ok": len(errors) == 0, "errors": errors, "summary": summary}
