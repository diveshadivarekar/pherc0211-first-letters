"""Tests for scoring ink maps against tifxyz-aligned ink labels."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location("eval_ink_labels", Path(__file__).resolve().parent.parent / "scripts" / "eval_ink_labels.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

ORIGIN = np.array([1000.0, 150.0, 300.0])
SHAPE = np.array([420.0, 120.0, 620.0])
H, W = 400, 600  # canvas pixels; one pixel = one voxel; surface plane y = 200, z = 1000 + row, x = 300 + col


def make_labels(tmp_path, unsupervised=True):
    """A flat segment: canvas (row, col) sits at (z, y, x) = (1000 + row, 200, 300 + col). Ink = columns 100-199 and 400-449."""
    d = tmp_path / "seg"
    d.mkdir()
    scale = 0.1
    gi, gj = np.mgrid[: int(H * scale) + 2, : int(W * scale) + 2].astype(np.float32)
    for name, arr in (("z", 1000 + gi / scale), ("y", np.full_like(gi, 200.0)), ("x", 300 + gj / scale)):
        Image.fromarray(arr.astype(np.float32), mode="F").save(d / f"{name}.tif")
    ink = np.zeros((H, W), np.uint8)
    ink[:, 100:200] = 255
    ink[:, 400:450] = 255
    sup = np.full((H, W), 255, np.uint8)
    if unsupervised:
        sup[:, 250:300] = 0
    Image.fromarray(ink).save(d / "seg_inklabels.tif")
    Image.fromarray(sup).save(d / "seg_supervision_mask.tif")
    (d / "meta.json").write_text(json.dumps({"scale": [scale, scale], "format": "tifxyz"}))
    return d


def make_ink_h5(path, y_offset=0.0, prob_of_ink=0.9, flip=False):
    """One page lying on the segment (optionally shifted in y): high probability exactly where the labels say ink."""
    rr, cc = np.mgrid[:H, :W]
    cm = np.stack([rr + 0.0, np.full((H, W), 50.0 + y_offset), cc + 0.0], axis=-1).astype(np.float32)  # tile-local (origin 1000,150,300)
    ink = ((cc >= 100) & (cc < 200)) | ((cc >= 400) & (cc < 450))
    prob = np.where(ink ^ flip, prob_of_ink, 0.05).astype(np.float32)
    with h5py.File(path, "w") as f:
        g = f.create_group("layer_1")
        g.create_dataset("coord_map", data=cm)
        g.create_dataset("ink_probability", data=prob)
    return path


def test_label_pixels_are_placed_in_3d_and_unsupervised_ones_dropped(tmp_path):
    lp = ev.label_points(make_labels(tmp_path), "seg", ORIGIN, SHAPE)
    n_sup = H * (W - 50)
    assert len(lp["ink"]) == n_sup and lp["ink"].sum() == H * 150
    z, y, x = lp["xyz_local"].T
    assert np.allclose(y, 50.0, atol=1e-3)  # plane y = 200 -> tile-local 50
    r, c = lp["rowcol"].T
    assert np.allclose(z, r, atol=0.51) and np.allclose(x, c, atol=0.51)  # canvas pixel -> its 3D position (to grid-interpolation accuracy)
    assert not ((c >= 250) & (c < 300)).any()


def test_points_outside_the_tile_are_excluded(tmp_path):
    small = np.array([200.0, 120.0, 300.0])  # z 1000-1200 and x 300-600 only
    lp = ev.label_points(make_labels(tmp_path), "seg", ORIGIN, small)
    z, y, x = lp["xyz_local"].T
    assert z.max() < 200 and x.max() < 300 and len(lp["ink"]) > 0


def test_metrics_perfect_inverted_and_uninformative_scores():
    rng = np.random.default_rng(0)
    ink = rng.random(4000) < 0.3
    perfect = np.where(ink, 0.9, 0.1)
    m = ev.metrics(perfect, ink)
    assert m["auc"] == pytest.approx(1.0) and m["f0.5"] == pytest.approx(1.0) and m["coverage"] == 1.0
    assert ev.metrics(1 - perfect, ink)["auc"] == pytest.approx(0.0)
    noise = ev.metrics(rng.random(4000), ink)
    assert abs(noise["auc"] - 0.5) < 0.05 and noise["f0.5"] < 0.5
    assert m["f0.5_all_ink"] == pytest.approx(1.25 * ink.mean() / (0.25 * ink.mean() + 1))


def test_metrics_report_coverage_and_use_only_covered_points():
    ink = np.array([True, True, False, False, True, False])
    score = np.array([0.9, 0.8, 0.1, 0.2, np.nan, np.nan])
    m = ev.metrics(score, ink)
    assert m["covered"] == 4 and m["coverage"] == pytest.approx(4 / 6) and m["auc"] == pytest.approx(1.0)


def test_scoring_through_pages_respects_the_distance_gate(tmp_path):
    lp = ev.label_points(make_labels(tmp_path), "seg", ORIGIN, SHAPE)
    on = ev.metrics(ev.score_from_pages(make_ink_h5(tmp_path / "on.h5"), lp, dmax=4.0), lp["ink"])
    assert on["coverage"] > 0.99 and on["auc"] == pytest.approx(1.0) and on["f0.5"] > 0.99
    off = ev.metrics(ev.score_from_pages(make_ink_h5(tmp_path / "off.h5", y_offset=10.0), lp, dmax=4.0), lp["ink"])
    assert off["coverage"] == 0.0  # a page 10 voxels away is not the labelled sheet: nothing is scored
    wide = ev.metrics(ev.score_from_pages(make_ink_h5(tmp_path / "off2.h5", y_offset=10.0), lp, dmax=12.0), lp["ink"])
    assert wide["coverage"] > 0.99 and wide["auc"] == pytest.approx(1.0)


def test_a_map_that_is_wrong_everywhere_scores_below_chance(tmp_path):
    lp = ev.label_points(make_labels(tmp_path), "seg", ORIGIN, SHAPE)
    m = ev.metrics(ev.score_from_pages(make_ink_h5(tmp_path / "bad.h5", flip=True), lp, dmax=4.0), lp["ink"])
    assert m["auc"] < 0.05
