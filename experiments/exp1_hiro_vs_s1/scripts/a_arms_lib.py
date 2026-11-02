"""Group A bookkeeping: which forcing stores exist, how ACE's Julian time
axis becomes the standard calendar DiffHydro reads, and the QM hash every
corrected arm carries (CLAUDE.md, Phase 3).

Members (CLAUDE.md, Phase 3):
- A-S1, A-coarse-raw: one per ACE run, ID = the run (`ic0000`);
- A-HiRO, A-HiRO-raw: one per HiRO sample, ID = `<run>-<sample>` (`ic0000-2`).
Every member's temperature is its parent run's QM'd temperature (D2, D17).
"""
import hashlib
import json
from dataclasses import dataclass

import cftime
import numpy as np
import pandas as pd
import xarray as xr

ARMS = {
    # arm: (precipitation source, precipitation QM'd?)
    "A-S1": ("ace2s", True),
    "A-HiRO": ("hiro", True),
    "A-HiRO-raw": ("hiro", False),
    "A-coarse-raw": ("ace2s", False),
}


@dataclass(frozen=True)
class Member:
    arm: str
    member: str
    run: str            # parent ACE run
    sample: int | None  # HiRO sample, None for ACE2S-based arms

    @property
    def source(self):
        return ARMS[self.arm][0]

    @property
    def qm_precip(self):
        return ARMS[self.arm][1]

    def store(self):
        return f"{self.arm}/{self.member}.zarr"


def members(cfg, arms=tuple(ARMS)):
    out = []
    for arm in arms:
        for run in cfg["members"]["ace_runs"]:
            if ARMS[arm][0] == "ace2s":
                out.append(Member(arm, run, run, None))
            else:
                out += [Member(arm, f"{run}-{s}", run, s) for s in cfg["members"]["hiro_samples"]]
    return out


def qm_hash(cfg, var):
    """Hash of the qm.<var> config section: identical for every arm QM'd
    with the same settings (D8, U3.2)."""
    blob = json.dumps(cfg["qm"][var], sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def to_standard_calendar(da, dim="time"):
    """Relabel a daily Julian cftime axis with the same year-month-day in
    the standard calendar; values are untouched (U3.3).

    Only labels change, so the seasonal cycle stays on the same calendar
    months (the QM groups by them). Valid only where the Julian and standard
    calendars agree on leap days (1901-2099): anything that would drop,
    duplicate or skip a day raises.
    """
    t = da[dim].values
    if not all(isinstance(x, cftime.DatetimeJulian) for x in t):
        raise ValueError(f"{dim} must be a Julian cftime axis")
    if any((x.hour, x.minute, x.second) != (0, 0, 0) for x in t):
        raise ValueError("expected daily labels at 00:00")
    labels = pd.to_datetime([f"{x.year:04d}-{x.month:02d}-{x.day:02d}" for x in t])
    if not (np.diff(labels.values) == np.timedelta64(1, "D")).all():
        raise ValueError("relabelled axis is not contiguous daily (leap-day mismatch)")
    return da.assign_coords({dim: labels})
