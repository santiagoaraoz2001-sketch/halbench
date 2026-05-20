# HalBench

A behavioral benchmark for **sycophancy and hallucination** in LLMs. The corpus consists of prompts built on a false premise (a fabricated reference, an overstated scope, an authority misapplied, an unanswerable question). When a model produces the requested content anyway, that single response is *both* failures at once: sycophantic (it agreed with a framing it should have flagged) *and* hallucinatory (the produced content elaborates on something that doesn't exist or can't be answered). HalBench reports one number per response that captures *how much false-premise content the model produced under social pressure*. Continuous scoring on `0 → 1`. Higher means more honest pushback; lower means more agreement and more fabrication.

- **Live Space + leaderboard**: https://huggingface.co/spaces/Specific-Labs/halbench
- **Dataset (corpus + responses + scores)**: https://huggingface.co/datasets/Specific-Labs/halbench
- **Python package**: `pip install halbench`

## What's in this repo

| Path | Contents |
|---|---|
| `halbench-pkg/` | The runnable Python package. CLI to run any OpenRouter-accessible model on the corpus, score responses, and produce a leaderboard submission. |
| `space/` | Source for the public Gradio Space (HF Space mirrors this directory). |
| `corpus/items/` | 3,200 prompt items, 100 per cell-field, across 32 cell-fields (8 mechanisms × 4 domains). |
| `corpus/cross_author_paragraph_anchors.json` | DEFER / SOFT / HARD reference paragraphs by a four-model panel. Used to calibrate endpoints. |
| `corpus/calibration_endpoints.json` | Locked per-cell-field DEFER / HARD endpoints. |
| `corpus/anchor_library.json` | Generic stance anchors used to define the M5 axis. |

## Quick start: score a new model

```bash
pip install halbench
export OPENROUTER_API_KEY="sk-or-v1-..."
export HF_TOKEN="hf_..."                    # for embedding inference

halbench run \
  --model "anthropic/claude-haiku-4.5" \
  --backend openrouter \
  --out ./runs/haiku-4.5
```

Output goes to `./runs/haiku-4.5/`:
- `responses.jsonl` — 3,200 chat completions
- `scores.jsonl` — per-response scores
- `report.md` — model summary

Full CLI docs: `halbench --help`.

## The grid

8 mechanisms × 4 domains = 32 cell-fields, 100 items each.

**Mechanisms**: `A1` Non-Existent Referent, `A2` False Attribute of Real Referent, `A3` Unanswerable by Construction, `B2` Epistemic-Authority Framing, `B3` Manufactured-Stakes Framing, `B4` Social-Consensus Framing, `C3` Confidence Coercion, `D1` Adversarial-Argument Confabulation.

**Domains**: `SK` Specialized Knowledge, `PC` Programming, `GK` General Knowledge, `GC` General Conversational.

Item IDs follow `{cell}_{field}__synth_{####}` (e.g. `B2_GC__synth_0015`).

## Scoring in one paragraph

Each response is split into sentences, embedded with [`microsoft/harrier-oss-v1-0.6b`](https://huggingface.co/microsoft/harrier-oss-v1-0.6b), and projected onto the M5 axis (the difference vector between embeddings of *"no"* and *"yes"*). A per-cell-field DEFER / HARD baseline computed from a four-model panel normalizes each per-sentence projection into `[0, 1]`. The arithmetic mean across sentences is the final score. The pipeline is deterministic; per-sentence vectors are preserved so any score can be inspected.

Full derivation, ablations, validation: [Methodology tab on the Space](https://huggingface.co/spaces/Specific-Labs/halbench).

## Human validation

Stratified 100-item sample read by a single human annotator. Mean Kendall τ = 0.43, 95% CI [0.35, 0.52], p < 10⁻⁷. Report: `halbench-pkg/leaderboard/README.md`.

## What HalBench is not

This is **not a safety benchmark**. It does not measure whether a model refuses harmful requests, weapons questions, self-harm content, or illegal activity. Every corpus prompt uses a non-harmful substrate (workplace ergonomics, garden clubs, hobby etiquette, programming style, consumer warranty terms) and the user in every prompt is *wrong*, not *malicious*. A model can score well here and still fail safety evals, and the reverse holds. HalBench is one input to model selection, not a substitute for a safety review. The full argument is on the Methodology tab.

## Citation

```bibtex
@misc{halbench2026,
  title  = {HalBench: a sycophancy and hallucination benchmark for LLMs},
  author = {Specific Labs},
  year   = {2026},
  url    = {https://github.com/santiagoaraoz2001-sketch/halbench},
}
```

## License

MIT.
