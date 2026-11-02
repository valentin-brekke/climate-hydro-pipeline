"""Tests for a_arms_lib (CLAUDE.md, Phase 3: U3.1-U3.3). The real-data
versions of U3.1 and U3.2 (on the written stores) run in
run_validate_a_arms.py."""
import numpy as np
import pytest
import xarray as xr

import a_arms_lib as aa

CFG = {"members": {"ace_runs": ["ic0000", "ic0001"], "hiro_samples": [0, 1, 2, 3]},
       "qm": {"precip": {"kind": "*", "nquantiles": 100}, "temp": {"kind": "+", "nquantiles": 100}}}


def _julian(start="2014-01-02", periods=3650):
    t = xr.date_range(start, periods=periods, freq="D", calendar="julian", use_cftime=True)
    return xr.DataArray(np.arange(periods, dtype="float32"), dims="time", coords={"time": t})


# --------------------------------------------------------------------------
# U3.1  member bookkeeping and temperature pairing
# --------------------------------------------------------------------------

def test_u3_1_members_and_parent_runs():
    """2 A-S1, 8 A-HiRO, 8 A-HiRO-raw and 2 A-coarse-raw members; every
    HiRO member's parent run is the run in its ID, so its temperature is
    that run's (the store-level equality is checked on real data)."""
    ms = aa.members(CFG)
    count = {a: sum(m.arm == a for m in ms) for a in aa.ARMS}
    assert count == {"A-S1": 2, "A-HiRO": 8, "A-HiRO-raw": 8, "A-coarse-raw": 2}
    for m in ms:
        assert m.member.split("-")[0] == m.run
        assert (m.sample is None) == (m.source == "ace2s")
    assert [m.member for m in ms if m.arm == "A-HiRO"][:5] == \
        ["ic0000-0", "ic0000-1", "ic0000-2", "ic0000-3", "ic0001-0"]
    assert len({m.store() for m in ms}) == len(ms)
    assert {m.arm for m in ms if m.qm_precip} == {"A-S1", "A-HiRO"}


# --------------------------------------------------------------------------
# U3.2  one QM hash per variable
# --------------------------------------------------------------------------

def test_u3_2_qm_hash_is_stable_and_specific():
    reordered = {"qm": {"precip": {"nquantiles": 100, "kind": "*"}, "temp": CFG["qm"]["temp"]}}
    assert aa.qm_hash(CFG, "precip") == aa.qm_hash(reordered, "precip")
    changed = {"qm": {"precip": {"kind": "*", "nquantiles": 50}}}
    assert aa.qm_hash(CFG, "precip") != aa.qm_hash(changed, "precip")
    assert aa.qm_hash(CFG, "precip") != aa.qm_hash(CFG, "temp")


# --------------------------------------------------------------------------
# U3.3  Julian -> standard calendar changes labels only
# --------------------------------------------------------------------------

def test_u3_3_calendar_conversion_keeps_values_and_dates():
    da = _julian()
    out = aa.to_standard_calendar(da)
    np.testing.assert_array_equal(out.values, da.values)
    assert out.time.dtype.kind == "M"
    assert [str(x)[:10] for x in out.time.values[[0, 789, -1]]] == \
        [f"{x.year:04d}-{x.month:02d}-{x.day:02d}" for x in da.time.values[[0, 789, -1]]]
    assert str(out.time.values[-1])[:10] == "2023-12-30"
    assert (out.time.dt.month.values == np.array([x.month for x in da.time.values])).all()
    assert "2016-02-29" in {str(x)[:10] for x in out.time.values}


def test_u3_3_calendar_conversion_rejects_what_it_cannot_map():
    with pytest.raises(ValueError):       # Julian 2100-02-29 has no standard-calendar twin
        aa.to_standard_calendar(_julian("2100-02-27", 5))
    noleap = xr.date_range("2014-01-02", periods=10, freq="D", calendar="noleap", use_cftime=True)
    with pytest.raises(ValueError):
        aa.to_standard_calendar(_julian(periods=10).assign_coords(time=noleap))
    six_hourly = xr.date_range("2014-01-02", periods=8, freq="6h", calendar="julian", use_cftime=True)
    with pytest.raises(ValueError):
        aa.to_standard_calendar(_julian(periods=8).assign_coords(time=six_hourly))
