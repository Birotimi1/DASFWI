"""run_field_das.py: tag uniqueness and band/budget helpers.

The field driver had NONE of our method until 2026-08-03 -- no switch, no
multiscale, no Route B starter -- so every knob here is new, and every new knob
is a chance to reintroduce the tag collision that has silently overwritten
results FIVE times in this project.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hpc/standalone"))
from run_field_das import parse_bands, allocate_iters, ARMS, SOLO_ARMS  # noqa: E402


def _tag(well="78A-32", arm="gc", refiner="gc", robust="envelope",
         optimizer="adam", starting="gradient", bands=None,
         iter_alloc="final-heavy", z_air=0.0, grad_smooth="none", smoke=False):
    """Mirror of the driver's tag construction (kept in step by the tests)."""
    if arm in SOLO_ARMS:
        refiner = arm
    pair = "" if refiner in ("l2", arm) else f"-{refiner}"
    rb = "" if robust == "envelope" or arm in SOLO_ARMS else f"+{robust}"
    return ("field_" + well + "_" + arm + pair + rb + "_" + optimizer
            + "_" + starting
            + ("_b" + bands.replace(",", "-") if bands else "")
            + ("_fh" if bands and iter_alloc == "final-heavy" else "")
            + ("_air" if z_air > 0 else "")
            + ("_g" if grad_smooth != "none" else "")
            + ("_smoke" if smoke else ""))


def test_every_knob_reaches_the_tag():
    """Each knob must change the tag, or two experiments share a directory and
    the second silently overwrites the first."""
    base = _tag()
    for kw in (dict(arm="switch"), dict(optimizer="sgd"),
               dict(starting="route_b"), dict(bands="5,8,full"),
               dict(z_air=100.0), dict(grad_smooth="wavelength"),
               dict(well="78B-32"), dict(smoke=True)):
        assert _tag(**kw) != base, f"{kw} does not reach the tag"


def test_robust_reaches_the_tag_for_arms_that_evaluate_it():
    """The exact collision found in run_switch.py: --robust was missing, so
    `switch --robust tfphase` and plain `switch` wrote to one directory."""
    assert _tag(arm="switch", robust="tfphase") != _tag(arm="switch")
    assert "+tfphase" in _tag(arm="switch", robust="tfphase")


def test_robust_is_NOT_tagged_for_solo_arms():
    """A solo arm pins lambda=0, so the robust slot never runs -- tagging it
    would split IDENTICAL results across two directories, the opposite error."""
    assert _tag(arm="gc", robust="tfphase") == _tag(arm="gc", robust="envelope")
    assert _tag(arm="convsi", robust="tfphase") == _tag(arm="convsi")


def test_solo_arm_is_its_own_refiner():
    """Otherwise lambda=0 would select the DEFAULT refiner and a cell labelled
    `convsi` would quietly be measuring gc."""
    assert "-" not in _tag(arm="convsi").split("_adam")[0].replace("78A-32", "")
    assert _tag(arm="tfphase") != _tag(arm="gc")


def test_multiscale_knobs_are_distinguishable():
    assert _tag(bands="5,8,full", iter_alloc="equal") != \
           _tag(bands="5,8,full", iter_alloc="final-heavy")
    assert _tag(bands="5,8,full") != _tag(bands="5,12,full")


def test_a_realistic_campaign_has_no_duplicate_tags():
    seen = {}
    for arm in ("switch", "gc", "convsi", "l2", "tfphase"):
        for start in ("route_b", "traveltime"):
            for bands in (None, "5,8,12,full"):
                t = _tag(arm=arm, starting=start, bands=bands)
                assert t not in seen, f"COLLISION {t}: {seen[t]} vs {(arm,start,bands)}"
                seen[t] = (arm, start, bands)
    assert len(seen) == 5 * 2 * 2


# --------------------------------------------------------------------------- #
def test_parse_bands():
    assert parse_bands("5,8,full") == [5.0, 8.0, None]
    assert parse_bands("full") == [None]
    with pytest.raises(ValueError):
        parse_bands("")


def test_allocation_preserves_budget_and_favours_the_top_band():
    for n in (1, 2, 3, 4):
        eq = allocate_iters(50, n, "equal")
        fh = allocate_iters(50, n, "final-heavy")
        assert sum(eq) == sum(fh) == 50 * n
        assert all(i > 0 for i in fh)
    fh = allocate_iters(50, 4, "final-heavy")
    assert fh[-1] == max(fh) and fh[-1] / sum(fh) == pytest.approx(0.5)


def test_arms_cover_the_field_misfits_we_can_actually_use():
    """l2 inherits the assumed-wavelet error at FORGE; convsi is
    source-independent. Both must be reachable so the choice can be TESTED."""
    for m in ("convsi", "gc", "tfphase", "l2", "switch"):
        assert m in ARMS
    assert "convsi" in SOLO_ARMS and "switch" not in SOLO_ARMS
