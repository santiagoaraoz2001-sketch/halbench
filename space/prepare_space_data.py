#!/usr/bin/env python3
"""Consolidate corpus + responses + scores + anchors into a single tidy
data bundle that the Gradio Space loads at startup.

Outputs (all under benchmark/space/data/):
  leaderboard.json         — summary stats per model
  breakdown.json           — per-cell + per-field + per-cell-field stats (with p-values)
  items.jsonl              — one row per corpus item: prompt + metadata
  responses_scores.jsonl   — one row per (item, model): response_text + score + raw_M5
  anchors.json             — frontier paragraph anchors per cell-field
  endpoints.json           — production calibration endpoints
  meta.json                — version, generation timestamp, model panel
"""
import os, sys, json, time
from collections import defaultdict
import numpy as np

ROOT = "/Users/santiagoaraoz/Desktop/Specific Labs/P6_Hallucination/Benchmarks/halbench_v2"
CORPUS = f"{ROOT}/benchmark/corpus"
RESULTS = f"{ROOT}/benchmark/results"
OUT = f"{ROOT}/benchmark/space/data"

MODELS = [
    ("sonnet-4.6", "anthropic__claude-sonnet-4.6", "anthropic/claude-sonnet-4.6"),
    ("gpt-5.4",    "openai__gpt-5.4",              "openai/gpt-5.4"),
    ("gemini-3.1", "google__gemini-3.1-pro-preview","google/gemini-3.1-pro-preview"),
    ("grok-4.3",   "x-ai__grok-4.3",               "x-ai/grok-4.3"),
]


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 80)
    print(" Preparing Space data bundle")
    print("=" * 80)

    # ---- 1. items.jsonl ----
    # Load every corpus item file (lightweight: id, cell, field, prompt, construct_name, domain)
    item_files = sorted(os.listdir(f"{CORPUS}/items"))
    print(f"\n[1/6] Building items.jsonl from {len(item_files)} corpus files...")
    items = []
    items_by_id = {}
    for fn in item_files:
        if not fn.endswith(".json"): continue
        d = json.load(open(f"{CORPUS}/items/{fn}"))
        row = {
            "item_id": d["item_id"],
            "cell": d["cell"],
            "field": d["field"],
            "cell_field": d["cell_field"],
            "construct_name": d.get("construct_name", ""),
            "field_name": d.get("field_name", ""),
            "domain": d.get("domain", ""),
            "prompt": d.get("prompt", ""),
            "elaboration_ask": d.get("elaboration_ask", ""),
            "construct_bearing_element": d.get("construct_bearing_element", ""),
            "substrate": d.get("substrate", ""),
        }
        items.append(row)
        items_by_id[row["item_id"]] = row
    with open(f"{OUT}/items.jsonl", "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")
    print(f"     wrote {len(items)} items")

    # ---- 2. responses_scores.jsonl ----
    # For each (item, model): response_text + score + raw_M5 + length + flags
    print(f"\n[2/6] Building responses_scores.jsonl...")
    n_rows = 0
    with open(f"{OUT}/responses_scores.jsonl", "w") as out_f:
        for short, dirname, full_id in MODELS:
            scores_p = f"{RESULTS}/{dirname}/scores.jsonl"
            resp_p   = f"{RESULTS}/{dirname}/responses.jsonl"
            if not (os.path.exists(scores_p) and os.path.exists(resp_p)):
                print(f"   ! missing files for {short}, skipping")
                continue
            scores = {r["item_id"]: r for r in load_jsonl(scores_p)}
            resps  = {r["item_id"]: r for r in load_jsonl(resp_p)}
            for iid, s in scores.items():
                r = resps.get(iid, {})
                row = {
                    "item_id": iid,
                    "model_short": short,
                    "model_id": full_id,
                    "cell": s["cell"],
                    "field": s["field"],
                    "cell_field": s["cell_field"],
                    "response_text": r.get("response_text", "") or "",
                    "response_len_words": s.get("response_len_words", 0),
                    "n_sentences": s.get("n_sentences", 0),
                    "score": s.get("score"),
                    "score_with_n1_floor": s.get("score_with_n1_floor"),
                    "is_n1_response": s.get("is_n1_response", False),
                    "raw_M5": s.get("raw_M5"),
                    "is_anomaly": s.get("is_anomaly", False),
                    "finish_reason": r.get("finish_reason", ""),
                    "latency_s": r.get("latency_s", 0),
                }
                out_f.write(json.dumps(row) + "\n")
                n_rows += 1
    print(f"     wrote {n_rows} (item, model) rows")

    # ---- 3. leaderboard.json ----
    # Per-model summary (mean, p50, p90, %>thresholds, anomaly count)
    print(f"\n[3/6] Building leaderboard.json...")
    leaderboard = []
    for short, dirname, full_id in MODELS:
        scores_p = f"{RESULTS}/{dirname}/scores.jsonl"
        if not os.path.exists(scores_p): continue
        rows = load_jsonl(scores_p)
        valid = [r["score"] for r in rows if r.get("score") is not None and not r.get("is_anomaly")]
        anomalies = sum(1 for r in rows if r.get("is_anomaly"))
        arr = np.array(valid)
        # v2.2.1: also compute the n=1-floored aggregate
        floored = [r["score_with_n1_floor"] for r in rows
                   if r.get("score_with_n1_floor") is not None and not r.get("is_anomaly")]
        n1_count = sum(1 for r in rows if r.get("is_n1_response"))
        flr_arr = np.array(floored) if floored else arr
        leaderboard.append({
            "model_short": short,
            "model_id": full_id,
            "n": len(valid),
            "n_anomalies": anomalies,
            "n_n1": n1_count,
            "mean": float(arr.mean()),
            "mean_with_n1_floor": float(flr_arr.mean()),
            "sd":   float(arr.std()),
            "p10":  float(np.percentile(arr, 10)),
            "p50":  float(np.percentile(arr, 50)),
            "p90":  float(np.percentile(arr, 90)),
            "pct_above_0.5": float((arr > 0.5).mean()),
            "pct_above_0.7": float((arr > 0.7).mean()),
            "pct_above_0.8": float((arr > 0.8).mean()),
        })
    leaderboard.sort(key=lambda r: -r["mean"])
    with open(f"{OUT}/leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"     wrote {len(leaderboard)} model rows")

    # ---- 4. breakdown.json ----
    # Reuse existing LEADERBOARD_BREAKDOWN.json
    print(f"\n[4/6] Copying breakdown.json...")
    src = f"{RESULTS}/LEADERBOARD_BREAKDOWN.json"
    if os.path.exists(src):
        with open(src) as f: bd = json.load(f)
        with open(f"{OUT}/breakdown.json", "w") as f: json.dump(bd, f, indent=2)
        print(f"     copied (per_cell={len(bd.get('per_cell',{}))}, per_field={len(bd.get('per_field',{}))}, per_cf={len(bd.get('per_cell_field',{}))})")
    else:
        print(f"     ! missing {src}, skipping")

    # ---- 5. anchors.json ----
    # Frontier-only paragraph anchors per cell-field for the Anchor Library tab
    print(f"\n[5/6] Building anchors.json (frontier-only)...")
    FRONTIER = {"anthropic/claude-sonnet-4.6", "google/gemini-3.1-pro-preview",
                "x-ai/grok-4.3", "openai/gpt-5.5"}
    doc = json.load(open(f"{CORPUS}/cross_author_paragraph_anchors.json"))
    by_cf = defaultdict(lambda: {"defer": [], "soft": [], "hard": []})
    n_kept = 0
    for it in doc["items"]:
        cf = it["item_id"].split("__")[0]
        for aid, stances in it.get("authors", {}).items():
            if aid not in FRONTIER: continue
            for s in ("defer", "soft", "hard"):
                t = stances.get(s)
                if t and len(t) > 30:
                    by_cf[cf][s].append({"author": aid, "text": t, "item_id": it["item_id"]})
                    n_kept += 1
    out_anchors = {cf: by_cf[cf] for cf in sorted(by_cf)}
    with open(f"{OUT}/anchors.json", "w") as f:
        json.dump(out_anchors, f, indent=2)
    print(f"     wrote {n_kept} anchors across {len(out_anchors)} cell-fields")

    # ---- 6. endpoints.json + meta.json ----
    print(f"\n[6/6] Copying endpoints.json + writing meta.json...")
    src = f"{CORPUS}/calibration_endpoints.json"
    if os.path.exists(src):
        with open(src) as f: ep = json.load(f)
        with open(f"{OUT}/endpoints.json", "w") as f: json.dump(ep, f, indent=2)

    meta = {
        "benchmark_version": "v2.2.1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scoring_method": "sentence_level_harrier_mean",
        "axis": "M5 centered projection (yes / yes-but / no)",
        "embedding_model": "microsoft/harrier-oss-v1-0.6b (HF Inference)",
        "endpoint_panel": ["anthropic/claude-sonnet-4.6", "google/gemini-3.1-pro-preview",
                           "x-ai/grok-4.3", "openai/gpt-5.5"],
        "n_corpus_items": len(items),
        "human_validation": {
            "n_items": 100,
            "mean_kendall_tau": 0.431,
            "ci_lower": 0.346,
            "ci_upper": 0.517,
            "p_value": "< 1e-7",
            "report": "VALIDATION_100ITEM_REPORT.md",
        },
        "models_scored": [{"short": s, "id": full_id, "results_dir": d}
                          for s, d, full_id in MODELS],
        "cell_names": {
            "A1": "Non-Existent Referent",
            "A2": "False Attribute of Real Referent (Inferential)",
            "A3": "Unanswerable-by-Construction",
            "B2": "Epistemic-Authority Framing",
            "B3": "Manufactured-Stakes Framing",
            "B4": "Social-Consensus Framing",
            "C3": "Confidence Coercion",
            "D1": "Adversarial-Argument Confabulation",
        },
        "field_names": {
            "SK": "Specialized Knowledge",
            "PC": "Programming",
            "GK": "General Knowledge",
            "GC": "General Conversational",
        },
    }
    with open(f"{OUT}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Summary ----
    print(f"\n{'='*80}")
    print(f" DONE — Space data bundle at: {OUT}")
    print(f"{'='*80}")
    for fn in sorted(os.listdir(OUT)):
        sz = os.path.getsize(f"{OUT}/{fn}") / 1024
        print(f"  {fn:<32} {sz:>10.1f} KB")


if __name__ == "__main__":
    main()
