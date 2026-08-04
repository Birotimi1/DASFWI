"""Field acceptance criteria, verified against ANALYTICALLY KNOWN answers.

FORGE has no true model, so every metric we have used until now (SSIM, MAPE) is
unavailable. These are the replacements, and they must be trustworthy BEFORE the
field result exists -- a criterion invented after seeing the answer is not a
test. Each is checked against a case whose answer is known by construction.
"""
import numpy as np
import pytest

from inversion.field_acceptance import (ACCEPT, arrival_lags, mismatch_rms,
                                        mismatch_reduction, cross_validate,
                                        compare_to_log, zone_boundaries,
                                        boundary_alignment)

DT, NT, NCH, F0 = 0.001, 800, 12, 20.0


def _gather(shift=0.0, n_ch=NCH, dead=()):
    t = np.arange(NT) * DT
    g = np.zeros((1, NT, n_ch))
    for c in range(n_ch):
        if c in dead:
            continue
        tt = t - (0.25 + shift + 0.004 * c)
        g[0, :, c] = (1 - 2 * (np.pi * F0 * tt) ** 2) * np.exp(-(np.pi * F0 * tt) ** 2)
    return g


# --------------------------------------------------------------------------- #
# 1. first-arrival mismatch -- Park's 51.7% is the bar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shift", [0.010, -0.016, 0.030])
def test_lags_recover_a_known_time_shift(shift):
    """No picking: cross-correlation must return the shift we imposed."""
    L = arrival_lags(_gather(shift=shift), _gather(), DT)
    assert np.allclose(L[np.isfinite(L)], shift, atol=1.5 * DT)


def test_dead_traces_are_excluded_not_averaged_in():
    """A dead channel has no arrival to mismatch; folding its noise-driven lag
    into the score would flatter or damage the result arbitrarily."""
    obs = _gather(dead=(3, 7))
    L = arrival_lags(_gather(shift=0.01, dead=(3, 7)), obs, DT)
    assert np.isnan(L[0, 3]) and np.isnan(L[0, 7])
    assert np.isfinite(L[0, 0]) and np.isfinite(L[0, 5])
    assert np.isfinite(mismatch_rms(_gather(shift=0.01, dead=(3, 7)), obs, DT))


def test_reduction_matches_the_construction():
    """A model that halves the shift must report ~50% reduction."""
    obs = _gather()
    r = mismatch_reduction(_gather(shift=0.020), _gather(shift=0.010), obs, DT)
    assert r["reduction_pct"] == pytest.approx(50.0, abs=8.0)
    assert r["park_inv2_pct"] == 51.7
    assert r["beats_park"] is False                     # 50% < 51.7%
    better = mismatch_reduction(_gather(shift=0.030), _gather(shift=0.002),
                                obs, DT)
    assert better["beats_park"] is True


def test_a_perfect_match_reduces_fully():
    obs = _gather()
    r = mismatch_reduction(_gather(shift=0.02), obs.copy(), obs, DT)
    assert r["rms_final_s"] < 1.5 * DT
    assert r["reduction_pct"] > 90.0


# --------------------------------------------------------------------------- #
# 2. two-well cross-validation -- no truth needed
# --------------------------------------------------------------------------- #
def test_identical_models_agree_perfectly():
    a = np.random.default_rng(0).uniform(2000, 5000, (40, 60))
    c = cross_validate(a, a.copy())
    assert c["rms_diff"] == pytest.approx(0.0, abs=1e-9)
    assert c["rel_diff_pct"] == pytest.approx(0.0, abs=1e-9)


def test_disagreement_is_quantified_and_weightable():
    a = np.full((20, 30), 3000.0)
    b = a.copy(); b[:, 15:] = 3300.0            # 10% high on half the model
    c = cross_validate(a, b)
    assert c["max_diff"] == pytest.approx(300.0)
    assert 0 < c["rel_diff_pct"] < 10.0
    # weighting to the AGREEING half must report ~zero: comparing unilluminated
    # cells would measure the starting model, not the inversion
    w = np.zeros_like(a); w[:, :15] = 1.0
    assert cross_validate(a, b, weights=w)["rms_diff"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 3-4. well log + zone boundaries (ready for the 58-32 sonic we do not yet hold)
# --------------------------------------------------------------------------- #
def test_log_comparison_on_a_known_offset():
    z = np.arange(0, 2000, 10.0)
    true = 1500 + 1.5 * z
    got = compare_to_log(true + 200.0, z, z, true)
    assert got["bias"] == pytest.approx(200.0, abs=1.0)
    assert got["rms"] == pytest.approx(200.0, abs=1.0)
    assert got["corr"] == pytest.approx(1.0, abs=1e-6)


def test_log_is_resampled_to_the_MODEL_grid():
    """A sonic log is far finer than a 10 m FWI grid; scoring at log resolution
    would penalise the inversion for detail it cannot represent."""
    zm = np.arange(0, 1000, 50.0)                    # coarse model
    zl = np.arange(0, 1000, 0.5)                     # fine log
    vl = 2000 + 0.8 * zl
    got = compare_to_log(2000 + 0.8 * zm, zm, zl, vl)
    assert got["n"] == len(zm)                       # one sample per model cell
    assert got["rms"] < 1.0


def test_zone_boundaries_are_found_where_built():
    """Zones I/II/III: alluvium -> consolidated -> granitoid."""
    z = np.arange(0, 2000, 10.0)
    v = np.where(z < 450, 2000.0, np.where(z < 1100, 3500.0, 5700.0))
    got = zone_boundaries(v, z, n_zones=3)
    assert len(got) == 2
    assert min(abs(got[0] - 450), abs(got[1] - 450)) < 60
    assert min(abs(got[0] - 1100), abs(got[1] - 1100)) < 60


def test_boundary_alignment_reports_per_boundary_error():
    z = np.arange(0, 2000, 10.0)
    v = np.where(z < 500, 2000.0, np.where(z < 1150, 3500.0, 5700.0))
    got = boundary_alignment(v, z, expected_depths=[450.0, 1100.0], tol_m=100.0)
    assert got["n_matched"] == 2 and got["all_within_tol"]
    tight = boundary_alignment(v, z, expected_depths=[450.0, 1100.0], tol_m=10.0)
    assert not tight["all_within_tol"]


def test_thresholds_are_fixed_in_advance():
    """A threshold chosen after seeing the answer is not a test. These are set
    at 'comparable to the published work', not at whatever we achieve."""
    assert ACCEPT["mismatch_reduction_pct"] == 51.7      # Park INV2
    assert ACCEPT["cross_well_rel_diff_pct"] == 10.0
    assert ACCEPT["boundary_tol_m"] == 100.0
