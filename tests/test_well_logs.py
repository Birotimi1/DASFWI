"""FORGE 58-32 wireline logs -- the field's only ground truth.

Runs only when the LAS is present ($FORGE_DAS_DIR), so the suite still passes
on a machine without the data. These assertions are the ones that decide
whether the log can be used AT ALL: right log, right depths, near enough to the
2-D section, and a well vertical enough that measured depth IS true depth.
"""
import os
import numpy as np
import pytest

from forge.well_logs import (FT_M, PARK_RANGE_M, slowness_to_velocity,
                             poisson_check, _default_log_path, load_58_32)

_HAS_LAS = _default_log_path() is not None
needs_las = pytest.mark.skipif(not _HAS_LAS, reason="58-32 LAS not available")


def test_slowness_conversion_is_exact():
    """100 us/ft is a textbook 3048 m/s."""
    assert slowness_to_velocity(100.0) == pytest.approx(3048.0, rel=1e-9)
    assert np.isnan(slowness_to_velocity(0.0))          # never inf
    assert np.isnan(slowness_to_velocity(np.nan))


def test_poisson_check_flags_a_wrong_assumption():
    vp = np.full(100, 5000.0)
    assert poisson_check(vp, vp / np.sqrt(3))["assumption_ok"]
    assert not poisson_check(vp, vp / 3.0)["assumption_ok"]   # ratio 3.0
    assert poisson_check(np.array([np.nan]), np.array([np.nan]))["n"] == 0


@needs_las
def test_it_is_the_log_park_used():
    """Park: 'a wireline sonic log from 656 to 2307 m'. If our depths differ we
    are validating against a different curve than the published comparison."""
    L = load_58_32()
    z = L["z_m"]
    assert np.nanmin(z) == pytest.approx(PARK_RANGE_M[0], abs=2.0)
    assert np.nanmax(z) == pytest.approx(PARK_RANGE_M[1], abs=2.0)


@needs_las
def test_the_well_is_vertical_enough_that_MD_is_TVD():
    """A deviated well means measured depth is NOT true depth, and the log
    would be compared at the wrong depths. Judge the CORRECTION against the
    grid cell, not the raw departure -- 56 m of departure sounds large and is
    actually 0.7 m of depth error, 0.07 of a 10 m cell."""
    L = load_58_32()
    assert L["tvd_correction_m"] < 1.0                  # metres
    assert L["tvd_correction_m"] < 0.2 * 10.0           # << one FORGE cell


@needs_las
def test_velocities_are_physical_and_shear_is_present():
    """DTSM gives Vs -- ground truth for the part of our claim Park leave as
    future work."""
    L = load_58_32()
    vp, vs = L["vp"], L["vs"]
    assert vs is not None, "no DTSM: this run has no shear"
    for v, lo, hi in ((vp, 2000, 7500), (vs, 1000, 5000)):
        g = v[np.isfinite(v)]
        assert g.size > 1000
        assert lo < np.median(g) < hi
    m = np.isfinite(vp) & np.isfinite(vs)
    assert (vp[m] > vs[m]).all(), "Vp must exceed Vs everywhere"


@needs_las
def test_the_sqrt3_assumption_holds_at_this_site():
    """vs_from_vp and the elastic synthetic both assume Vp/Vs = sqrt(3). This
    log is the first chance to CHECK that rather than assume it."""
    L = load_58_32()
    pc = poisson_check(L["vp"], L["vs"])
    assert pc["n"] > 1000
    assert pc["median_ratio"] == pytest.approx(1.82, abs=0.10)
    assert pc["assumption_ok"], f"sqrt(3) refuted: measured {pc['median_ratio']:.3f}"


@needs_las
def test_the_log_cannot_validate_the_SHALLOW_section():
    """A limitation that must not be forgotten when reporting: the sonic starts
    at ~655 m, so it says NOTHING about the near surface -- which is exactly
    where the synthetic shows error growing 153 -> 348 m/s. The log validates
    where we are probably fine and is silent where we are probably worst."""
    L = load_58_32()
    assert np.nanmin(L["z_m"]) > 600.0
    assert np.nanmin(L["z_m"]) > 450.0     # deeper than the zone I/II boundary
