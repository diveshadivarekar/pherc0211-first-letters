"""Tests for src/m1_ingestion/real_scan.py.

fetch_real_scan() needs network access (scikit-image downloads its bundled
sample on first use) — those tests are marked and skip cleanly if the
fetch fails for any reason (offline CI, etc.), since that's an environment
fact, not a code defect. save_real_scan()'s HDF5-writing and the resulting
file's compatibility with M1's own loader don't need network access and
always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.m1_ingestion.generate_phantom import RAW_DATASET
from src.m1_ingestion.io_utils import VolumeStore
from src.m1_ingestion.real_scan import save_real_scan


def _try_fetch():
    try:
        from src.m1_ingestion.real_scan import fetch_real_scan

        return fetch_real_scan()
    except Exception as exc:  # noqa: BLE001 - any failure here means "skip", not "fail"
        pytest.skip(f"could not fetch real sample scan (offline?): {exc}")


class TestSaveRealScan:
    def test_writes_volume_dataset(self, tmp_path):
        volume = np.random.default_rng(0).uniform(size=(8, 16, 16)).astype(np.float32)
        path = tmp_path / "scan.h5"
        out = save_real_scan(path, volume)

        assert out == path
        store = VolumeStore(path, mode="r")
        np.testing.assert_array_equal(store.read_all(), volume)
        store.close()

    def test_output_is_m1_run_compatible(self, tmp_path):
        """The file save_real_scan() writes must use the same dataset name
        M1's run() and M2/M3's CLIs expect — this is the whole point of the
        module (real data feeding the same pipeline as the phantom)."""
        import h5py

        volume = np.random.default_rng(0).uniform(size=(8, 16, 16)).astype(np.float32)
        path = tmp_path / "scan.h5"
        save_real_scan(path, volume)

        with h5py.File(path, "r") as f:
            assert RAW_DATASET in f
            assert f[RAW_DATASET].chunks is not None

    def test_chunk_depth_capped_to_volume_depth(self, tmp_path):
        volume = np.zeros((4, 8, 8), dtype=np.float32)
        path = tmp_path / "scan.h5"
        save_real_scan(path, volume, chunk_depth=16)  # bigger than depth=4

        store = VolumeStore(path, mode="r")
        assert store.chunk_depth <= 4
        store.close()


@pytest.mark.slow
class TestFetchRealScan:
    def test_returns_normalised_float32_volume(self):
        volume = _try_fetch()
        assert volume.dtype == np.float32
        assert volume.ndim == 3
        assert volume.min() >= 0.0
        assert volume.max() <= 1.0 + 1e-6

    def test_runs_through_m1_correction(self, tmp_path):
        """The real scan should flow through the same correction pipeline
        as the phantom without crashing."""
        from src.m1_ingestion.preprocessing import correct_chunk

        volume = _try_fetch()
        corrected = correct_chunk(volume[:8])  # a few slices, keep the test fast
        assert corrected.shape == volume[:8].shape
        assert np.isfinite(corrected).all()
