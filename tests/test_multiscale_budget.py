"""Multiscale band ladder + iteration budget (hpc/marmousi_full_das/run_adaptive.py).

Phase B concluded "multiscale HURTS the switch" (0.464 vs 0.626). Two flaws in
the SETUP, not the method, are pinned here:

  1. the default ladder had a redundant band -- the source reaches only
     f90=6.25 Hz, so an explicit 6.25 AND `full` clamp to the same cutoff and
     invert identical data, burning a quarter of the budget;
  2. `--iters` was per band with an EQUAL split, so a 4-band cascade got 25% of
     its budget at the top band while the single-scale control it was compared
     against got 100% -- and the score is measured at the top band.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc/marmousi_full_das"))
from run_adaptive import allocate_iters, parse_bands, DEFAULT_BANDS   # noqa: E402


def _effective(bands, f90):
    return [(f90 if b is None else min(b, f90)) for b in bands]


def test_default_ladder_has_no_redundant_band():
    """The old default 3.0,4.5,6.25,full clamped two bands to the same 6.25 Hz."""
    f90 = 6.25                                   # integrated Ricker, F0=5, our grid
    eff = _effective(parse_bands(DEFAULT_BANDS), f90)
    assert len(set(eff)) == len(eff), f"redundant bands in the default: {eff}"
    assert eff == sorted(eff), "ladder must ascend"


def test_the_old_default_would_now_be_rejected():
    f90 = 6.25
    eff = _effective(parse_bands("3.0,4.5,6.25,full"), f90)
    assert len(set(eff)) < len(eff)              # exactly what preflight refuses


def test_allocation_preserves_the_total_budget():
    """final-heavy must REALLOCATE, never inflate -- otherwise the cascade would
    simply be buying its win with extra iterations."""
    for n in (2, 3, 4, 5):
        eq = allocate_iters(75, n, "equal")
        fh = allocate_iters(75, n, "final-heavy")
        assert sum(eq) == sum(fh) == 75 * n
        assert len(fh) == n and all(i > 0 for i in fh)


def test_final_heavy_favours_the_band_that_sets_the_score():
    fh = allocate_iters(75, 4, "final-heavy")
    assert fh[-1] == 150 and fh[-1] == max(fh)
    assert fh[-1] / sum(fh) == pytest.approx(0.5)      # 50%, was 25%
    # ...and still LESS than the single-scale control's 300 at full band, so the
    # comparison stays honest rather than tilted the other way.
    assert fh[-1] < sum(fh)


def test_equal_allocation_is_unchanged():
    assert allocate_iters(75, 4, "equal") == [75] * 4      # back-compatible


def test_unknown_allocation_is_rejected():
    with pytest.raises(ValueError):
        allocate_iters(75, 3, "no-such-mode")
