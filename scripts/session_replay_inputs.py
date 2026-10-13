"""Validate recorded aperture samples before constructing a replay command path."""

import numpy as np


def replay_widths(images, query_times, width_range, max_gap_s):
    """Interpolate only adjacent valid measurements; never bridge or extrapolate."""
    times = np.asarray([r['device_s'] for r in images], float)
    query = np.asarray(query_times, float)
    limits = np.asarray(width_range, float)
    if (len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Replay needs at least two strictly increasing image timestamps')
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError('Replay query timestamps must be finite and one-dimensional')
    if (limits.shape != (2,) or not np.isfinite(limits).all()
            or limits[0] < 0 or limits[1] <= limits[0]
            or not np.isfinite(max_gap_s) or max_gap_s <= 0):
        raise ValueError('Invalid measured aperture range or image-gap bound')
    widths = np.asarray([
        r['width_measurement'].get('opening_m')
        if r.get('width_measurement', {}).get('valid') else np.nan
        for r in images], float)
    good = np.isfinite(widths) & (widths >= limits[0]) & (widths <= limits[1])
    right = np.searchsorted(times, query, side='left')
    r = np.clip(right, 0, len(times)-1)
    left = np.maximum(r-1, 0)
    exact = (query == times[r]) & good[r]
    bracket = ((right > 0) & (right < len(times)) & good[left] & good[r]
               & (times[r]-times[left] <= max_gap_s))
    valid = exact | bracket
    if not valid.all():
        raise ValueError(
            f'Replay rejected: {int((~valid).sum())}/{len(query)} targets lack '
            'adjacent valid jaw-width measurements; inspect/recollect the episode. '
            'Missing widths are not filled or treated as open.')
    # All interpolation intervals have been validated above.
    return np.interp(query, times, widths)
