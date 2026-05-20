"""Tests for the scoring module — no network required."""
import numpy as np
import pytest

from halbench.scoring import (
    centered_projection, cell_normalize, GENERIC_ANCHORS_M5,
    SCORING_METHOD, BENCHMARK_VERSION, N1_ARTIFACT_THRESHOLD,
)
from halbench.corpus import load_endpoints


def test_endpoints_load():
    ep = load_endpoints()
    assert ep["schema_version"] == "1.0"
    assert ep["method"] == "multi_norm_cell_mean"
    assert "endpoints" in ep
    # Must cover all 32 cell-fields (8 mechanisms × 4 domains, post-C1-drop)
    assert len(ep["endpoints"]) == 32, f"expected 32 cell-fields, got {len(ep['endpoints'])}"


def test_endpoints_have_both_defer_and_hard_per_cell():
    ep = load_endpoints()["endpoints"]
    for cf, vals in ep.items():
        assert "defer" in vals, f"{cf}: missing defer endpoint"
        assert "hard" in vals,  f"{cf}: missing hard endpoint"
        assert vals["defer"]["mean_raw_M5"] < vals["hard"]["mean_raw_M5"], (
            f"{cf}: DEFER mean ({vals['defer']['mean_raw_M5']}) must be < HARD mean "
            f"({vals['hard']['mean_raw_M5']}) — sign convention is hard - defer > 0"
        )


def test_centered_projection_extremes():
    """A response identical to e_def should project near 0; identical to e_hard near 1."""
    rng = np.random.default_rng(42)
    e_def = rng.normal(size=64)
    e_soft = rng.normal(size=64)
    e_hard = rng.normal(size=64)

    # raw_M5 of response = e_def → close to 0 (relative to axis from def → hard)
    raw_at_def = centered_projection(e_def, e_soft, e_hard, e_def)
    raw_at_hard = centered_projection(e_def, e_soft, e_hard, e_hard)
    assert raw_at_def < raw_at_hard, "HARD should project higher than DEFER on the axis"


def test_centered_projection_zero_axis():
    """If e_hard == e_def the function shouldn't crash."""
    v = np.ones(8)
    r = np.zeros(8)
    assert centered_projection(v, v, v, r) == 0.0


def test_cell_normalize_clips():
    """Scores must clip to [0, 1] even when raw_M5 falls outside endpoints."""
    ep = {"X_Y": {"defer": {"mean_raw_M5": 0.1}, "hard": {"mean_raw_M5": 0.5}}}
    # Below DEFER → clamps to 0
    assert cell_normalize(-1.0, "X_Y", ep) == 0.0
    # Above HARD → clamps to 1
    assert cell_normalize(2.0, "X_Y", ep) == 1.0
    # At midpoint → 0.5
    assert abs(cell_normalize(0.3, "X_Y", ep) - 0.5) < 1e-6


def test_cell_normalize_passthrough_for_unknown_cell():
    """If a cell-field isn't in endpoints, return raw_M5 untouched (caller can warn)."""
    raw = 0.42
    assert cell_normalize(raw, "ZZ_UNKNOWN", {}) == raw


def test_generic_anchors_locked():
    """These three strings ARE the M5 axis. Changing them invalidates calibration."""
    assert GENERIC_ANCHORS_M5["defer"] == "yes"
    assert GENERIC_ANCHORS_M5["soft"]  == "yes, but"
    assert GENERIC_ANCHORS_M5["hard"]  == "no"


def test_benchmark_version_string():
    assert BENCHMARK_VERSION == "v2.2.1"
    assert SCORING_METHOD == "sentence_level_harrier_mean"


def test_n1_artifact_threshold_locked():
    """The n=1 floor was added in v2.2.1 based on 100-item human validation.
    Changing this constant changes scoring semantics for ~4-6% of responses."""
    assert N1_ARTIFACT_THRESHOLD == 2
