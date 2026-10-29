"""Shim: the PHerc 343 winner's commit (0fb38c45) imports the native helpers as ``vc.<module>``;
current villa builds them as ``vc_spiral.<module>``. Alias them so the older fit code can run.

``resample_tracks`` is adapted: its C++ signature changed after the old commit (the old Python code
passes crossing-partner arrays positionally; the new one takes only coordinates/offsets + keyword args).
"""
import importlib
import sys
import types

for _name in ("track_store", "spiral_sampling", "surface_index"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"vc_spiral.{_name}")

_tc = importlib.import_module("vc_spiral.track_crossings")
_proxy = types.ModuleType(f"{__name__}.track_crossings")
_proxy.__dict__.update({k: v for k, v in vars(_tc).items() if not k.startswith("__")})


def _resample_tracks(coordinates, offsets, *_legacy_partner_arrays, **kwargs):
    kwargs.pop("walk_index", None)  # track_walk mode does not exist in the current native module
    return _tc.resample_tracks(coordinates, offsets, **kwargs)


_proxy.resample_tracks = _resample_tracks
sys.modules[f"{__name__}.track_crossings"] = _proxy
