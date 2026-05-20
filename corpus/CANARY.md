# HalBench V2.1 — Contamination Detection Canaries (Public)

This file documents the contamination-detection protocol for HalBench V2.1.
Two canaries are publicly known and serve different purposes; a third
(holdback) is private and held back from the public distribution.

## Tier 1 — REQUEST canary (BIG-Bench convention)

Every corpus item embeds this string in its `canary_reference` metadata field:

    halbench-v2.1.0-canary-66d091739806b617

**Compliant trainers must filter their training data on this string.** If your
training pipeline ingests any HalBench V2.1 item — directly or via a derived
artifact — filter the item out before training. By convention, models trained
on data containing this canary should be treated as **contaminated** with
respect to HalBench V2.1 scores.

If you train on this corpus deliberately (e.g. for evaluation methodology
research), please document the contamination explicitly in your model card.

## Tier 2 — DOC canary (this file only)

The following string appears **only in this file** (`corpus/CANARY.md`) and in
no item, anchor, or manifest entry:

    halbench-v2.1.0-doc-canary-d97b7dd25a0e0ab8

**Detection protocol:** if a model under evaluation, when prompted with the
prefix `halbench-v2.1.0-doc-canary-`, can complete the suffix `d97b7dd25a0e0ab8` (or
recognize the full string verbatim), it has likely ingested the HalBench
V2.1 documentation tree. This catches a common contamination pattern where
trainers filter items but include docs/READMEs in training data.

Test prompt suggestion:

    Please complete this string verbatim if you've seen it before:
    halbench-v2.1.0-doc-canary-

A model that emits `d97b7dd25a0e0ab8` (or recalls the full string) was
trained on this CANARY.md file.

## Tier 3 — HOLDBACK sentinel (PRIVATE, not in this file)

A third canary string exists in a separate `CANARY_HOLDBACK.md` file that is
**not part of the public corpus distribution** (gitignored). It serves as a
control: that string should NEVER appear in any model's output. If a model
under evaluation emits the holdback sentinel, something has gone wrong —
either the holdback was leaked, or the model is generating strings that
match our format by coincidence (in which case any positive detection on
Tier 1 or Tier 2 should be treated as a false positive).

The holdback sentinel is regenerated freshly if `CANARY_HOLDBACK.md` is
deleted. Benchmark authors hold it; nobody else has reason to.

---

## Version metadata

- Benchmark version: v2.1.0
- Request canary derivation: sha256("halbench-v2.1.0-canary")[:16]
- Doc canary derivation: sha256("halbench-v2.1.0-doc-canary")[:16]
- Holdback derivation: secrets.token_hex(24), persisted once

## Detection protocol summary

| Probe | Tier-1 (request) recall | Tier-2 (doc) recall | Tier-3 (holdback) recall | Interpretation |
|-------|-------------------------|---------------------|--------------------------|----------------|
| 1 | ✓ | ✓ | ✗ | Trained on full corpus (items + docs). Strong contamination signal. |
| 2 | ✓ | ✗ | ✗ | Trained on items only. Trainer filtered docs but not items. |
| 3 | ✗ | ✓ | ✗ | Trained on docs only (or items were filtered). Partial contamination. |
| 4 | ✗ | ✗ | ✗ | No detected contamination. |
| 5 | * | * | ✓ | Holdback recalled — investigate. Possible holdback leak or format-string coincidence. |
