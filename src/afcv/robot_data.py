"""Timestamp alignment for the paper's asynchronously recorded scene cameras."""

import numpy as np


def nearest_timestamps(reference, candidate, max_offset=0.067):
    """Return nearest candidate indices and a validity mask, in seconds.

    Ties select the earlier frame. No interpolation, calibration, or action
    relabeling is performed. Each stream must be strictly time ordered.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    for values in (reference, candidate):
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Timestamps must be nonempty finite vectors")
        if np.any(np.diff(values) <= 0):
            raise ValueError("Timestamps must be strictly increasing")
    if max_offset < 0:
        raise ValueError("max_offset must be nonnegative")
    right = np.clip(np.searchsorted(candidate, reference), 0, len(candidate) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(candidate[left] - reference) <= abs(candidate[right] - reference), left, right)
    return indices, abs(candidate[indices] - reference) <= max_offset + 1e-12
