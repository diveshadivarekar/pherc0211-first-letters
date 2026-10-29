"""Tests for src/m1_ingestion/real_scroll.py.

Everything here needs network access to the public Vesuvius Challenge S3
bucket — marked @pytest.mark.slow and skips cleanly (not fails) if the
bucket is unreachable, since that's an environment fact, not a code defect.
Kept deliberately cheap: inspect_levels() only reads zarr metadata (no
volume data), and the download test pulls a handful of slices at the
smallest available resolution level, not a multi-GB crop.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.m1_ingestion.io_utils import VolumeStore

SCROLL_ID = "PHerc0332"


class _FakeZarrArray:
    """Enough of a zarr array's interface for download_scroll_crop: a
    fixed shape, a chunk grid (real buckets use 128-deep chunks; this uses
    a small one so tests stay fast), and slicing along axis 0. ``fail_until``
    simulates the real transient connection error a full run hit partway
    through — the first ``fail_until`` fetches raise, then it succeeds."""

    def __init__(self, shape, fail_until: int = 0, chunk_z: int = 4):
        self.shape = shape
        self.dtype = np.uint8
        self.chunks = (chunk_z, shape[1], shape[2])
        self._calls = 0
        self._fail_until = fail_until
        rng = np.random.default_rng(0)
        self._data = rng.integers(0, 256, size=shape, dtype=np.uint8)

    def __getitem__(self, key):
        self._calls += 1
        if self._calls <= self._fail_until:
            raise ConnectionError("simulated transient S3 connection drop")
        return self._data[key]


class TestFetchSlabWithRetry:
    def test_succeeds_immediately_when_no_failure(self):
        from src.m1_ingestion.real_scroll import _fetch_slab_with_retry

        arr = _FakeZarrArray((10, 4, 4))
        result = _fetch_slab_with_retry(arr, 0, 4, max_attempts=3, base_delay=0)
        np.testing.assert_array_equal(result, arr._data[0:4])

    def test_retries_past_transient_failures(self):
        from src.m1_ingestion.real_scroll import _fetch_slab_with_retry

        arr = _FakeZarrArray((10, 4, 4), fail_until=2)
        result = _fetch_slab_with_retry(arr, 0, 4, max_attempts=5, base_delay=0)
        np.testing.assert_array_equal(result, arr._data[0:4])
        assert arr._calls == 3

    def test_raises_after_exhausting_attempts(self):
        from src.m1_ingestion.real_scroll import _fetch_slab_with_retry

        arr = _FakeZarrArray((10, 4, 4), fail_until=10)
        with pytest.raises(ConnectionError):
            _fetch_slab_with_retry(arr, 0, 4, max_attempts=3, base_delay=0)


class TestDownloadScrollCropResume:
    """Unit tests for the resume/retry mechanics themselves, against a fake
    zarr array — no network, no real S3 bucket. Network-marked tests below
    cover the real bucket."""

    @pytest.fixture
    def patched(self, monkeypatch):
        fake_arr = _FakeZarrArray((20, 4, 4))

        def fake_find_path(scroll_id):
            return "fake://irrelevant"

        class FakeGroup(dict):
            pass

        fake_group = FakeGroup({"3": fake_arr})

        import src.m1_ingestion.real_scroll as real_scroll_module

        monkeypatch.setattr(real_scroll_module, "_find_volume_zarr_path", fake_find_path)
        monkeypatch.setattr("s3fs.S3FileSystem", lambda **kw: None)
        monkeypatch.setattr("s3fs.S3Map", lambda **kw: None)
        monkeypatch.setattr("zarr.open", lambda *a, **kw: fake_group)
        return fake_arr

    def test_fresh_download_writes_expected_shape(self, patched, tmp_path):
        from src.m1_ingestion.real_scroll import download_scroll_crop

        out_path = tmp_path / "crop.h5"
        download_scroll_crop(
            SCROLL_ID, out_path, level=3, z_fraction=0.5, z_start_fraction=0.0, stream_depth=4, progress=False
        )

        store = VolumeStore(out_path, mode="r")
        assert store.shape == (10, 4, 4)
        assert store.attrs["completed_slices"] == 10
        store.close()

    def test_resumes_from_partial_download(self, patched, tmp_path, monkeypatch):
        from src.m1_ingestion.real_scroll import download_scroll_crop

        out_path = tmp_path / "crop.h5"

        # Simulate a crash after the first slab by patching the fetch to
        # raise on the 2nd call — download_scroll_crop should still have
        # persisted (and flushed) the 1st slab with a completed_slices marker.
        import src.m1_ingestion.real_scroll as real_scroll_module

        call_count = {"n": 0}
        original_fetch = real_scroll_module._fetch_slab_with_retry

        def crash_after_one_slab(arr, z0, z1, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated crash")
            return original_fetch(arr, z0, z1, **kwargs)

        monkeypatch.setattr(real_scroll_module, "_fetch_slab_with_retry", crash_after_one_slab)

        with pytest.raises(RuntimeError):
            download_scroll_crop(
                SCROLL_ID, out_path, level=3, z_fraction=0.5, z_start_fraction=0.0, stream_depth=4, progress=False
            )

        store = VolumeStore(out_path, mode="r")
        assert store.attrs["completed_slices"] == 4  # first slab (stream_depth=4) committed
        store.close()

        # Resume: should pick up from slice 4, not restart from 0.
        monkeypatch.setattr(real_scroll_module, "_fetch_slab_with_retry", original_fetch)
        download_scroll_crop(
            SCROLL_ID, out_path, level=3, z_fraction=0.5, z_start_fraction=0.0, stream_depth=4, progress=False, resume=True
        )

        store = VolumeStore(out_path, mode="r")
        assert store.attrs["completed_slices"] == 10
        full = store.read_all()
        store.close()
        np.testing.assert_array_equal(full, (patched._data[0:10].astype(np.float32) / 255.0))

    def test_no_resume_forces_restart(self, patched, tmp_path):
        from src.m1_ingestion.real_scroll import download_scroll_crop

        out_path = tmp_path / "crop.h5"
        download_scroll_crop(
            SCROLL_ID, out_path, level=3, z_fraction=0.5, z_start_fraction=0.0, stream_depth=4, progress=False
        )
        # Re-run with resume=False: must not error even though the file already exists.
        download_scroll_crop(
            SCROLL_ID,
            out_path,
            level=3,
            z_fraction=0.5,
            z_start_fraction=0.0,
            stream_depth=4,
            progress=False,
            resume=False,
        )
        store = VolumeStore(out_path, mode="r")
        assert store.attrs["completed_slices"] == 10
        store.close()


def _skip_if_unreachable(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any failure here means "skip", not "fail"
        pytest.skip(f"Vesuvius Challenge open-data bucket unreachable: {exc}")


@pytest.mark.slow
class TestInspectLevels:
    def test_returns_multiscale_levels_smallest_first_not_required_but_present(self):
        from src.m1_ingestion.real_scroll import inspect_levels

        levels = _skip_if_unreachable(inspect_levels, SCROLL_ID)
        assert len(levels) >= 2
        for lvl in levels:
            assert lvl["shape"][0] > 0 and lvl["shape"][1] > 0 and lvl["shape"][2] > 0
            assert lvl["gb"] > 0

    def test_higher_level_number_is_smaller(self):
        """OME-Zarr convention: level 0 = full res, increasing level =
        more downsampled = smaller array."""
        from src.m1_ingestion.real_scroll import inspect_levels

        levels = _skip_if_unreachable(inspect_levels, SCROLL_ID)
        levels_sorted = sorted(levels, key=lambda l: l["level"])
        for a, b in zip(levels_sorted, levels_sorted[1:]):
            assert b["gb"] < a["gb"]


@pytest.mark.slow
class TestDownloadScrollCrop:
    def test_downloads_a_small_real_crop(self, tmp_path):
        from src.m1_ingestion.real_scroll import download_scroll_crop, inspect_levels

        levels = _skip_if_unreachable(inspect_levels, SCROLL_ID)
        smallest_level = max(l["level"] for l in levels)  # most downsampled = fastest test

        out_path = tmp_path / "crop.h5"
        result = _skip_if_unreachable(
            download_scroll_crop,
            SCROLL_ID,
            out_path,
            level=smallest_level,
            z_fraction=0.02,  # a handful of slices, not the whole level
            z_start_fraction=0.4,
            stream_depth=4,
            progress=False,
        )

        assert result == out_path
        store = VolumeStore(out_path, mode="r")
        volume = store.read_all()
        assert volume.dtype == np.float32
        assert volume.min() >= 0.0
        assert volume.max() <= 1.0 + 1e-6
        assert np.isfinite(volume).all()
        store.close()

    def test_output_flows_through_m1_correction(self, tmp_path):
        from src.m1_ingestion.preprocessing import correct_chunk
        from src.m1_ingestion.real_scroll import download_scroll_crop, inspect_levels

        levels = _skip_if_unreachable(inspect_levels, SCROLL_ID)
        smallest_level = max(l["level"] for l in levels)

        out_path = tmp_path / "crop.h5"
        _skip_if_unreachable(
            download_scroll_crop,
            SCROLL_ID,
            out_path,
            level=smallest_level,
            z_fraction=0.02,
            z_start_fraction=0.4,
            stream_depth=4,
            progress=False,
        )

        store = VolumeStore(out_path, mode="r")
        volume = store.read_all()
        store.close()

        corrected = correct_chunk(volume)
        assert corrected.shape == volume.shape
        assert np.isfinite(corrected).all()
