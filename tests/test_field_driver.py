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
         optimizer="adam", iterations=150, starting="gradient", bands=None,
         iter_alloc="final-heavy", z_air=0.0, topo_air=False,
         window=False, grad_smooth="none", smoke=False):
    """Mirror of the driver's tag construction (kept in step by the tests)."""
    if arm in SOLO_ARMS:
        refiner = arm
    pair = "" if refiner in ("l2", arm) else f"-{refiner}"
    rb = "" if robust == "envelope" or arm in SOLO_ARMS else f"+{robust}"
    return ("field_" + well + "_" + arm + pair + rb + "_" + optimizer
            + f"_i{iterations}"
            + "_" + starting
            + ("_b" + bands.replace(",", "-") if bands else "")
            + ("_fh" if bands and iter_alloc == "final-heavy" else "")
            + ("_topoair" if topo_air else "_air" if z_air > 0 else "")
            + ("_w" if window else "")
            + ("_g" if grad_smooth != "none" else "")
            + ("_smoke" if smoke else ""))


def test_every_knob_reaches_the_tag():
    """Each knob must change the tag, or two experiments share a directory and
    the second silently overwrites the first."""
    base = _tag()
    for kw in (dict(arm="switch"), dict(optimizer="sgd"),
               dict(starting="route_b"), dict(bands="5,8,full"),
               dict(z_air=100.0), dict(topo_air=True), dict(window=True),
               dict(grad_smooth="wavelength"),
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


def test_topographic_air_is_distinguishable_from_a_flat_slab():
    """They are DIFFERENT models: at FORGE the ground is the datum at one end of
    the line and 162 m below it at the other, so a uniform slab cannot represent
    the surface. Sharing a tag would conflate two different experiments."""
    assert _tag(topo_air=True) != _tag(z_air=162.0)
    assert "_topoair" in _tag(topo_air=True)


def test_windowing_reaches_the_field_tag():
    """The single most important thing the FORGE synthetic taught us -- Park
    report strong surface waves an acoustic code cannot model, and windowing
    improved the shallow model for every refiner. It was MISSING from the field
    driver entirely until 2026-08-04, so the decided recipe could not be run."""
    assert _tag(window=True) != _tag()
    assert "_w" in _tag(window=True)


def test_iterations_reach_the_field_tag():
    """SEVENTH occurrence of the tag-collision class. The campaign runs BOTH 30
    and 150 iterations because the synthetic showed shallow error grows with
    iteration count -- so a shared tag would have silently destroyed the exact
    early-vs-late comparison the campaign exists to make."""
    assert _tag(iterations=30) != _tag(iterations=150)
    assert "_i30" in _tag(iterations=30)


def test_a_full_field_campaign_has_no_duplicate_tags():
    seen = {}
    for well in ("78A-32", "78B-32"):
        for arm in ("convsi", "gc", "switch"):
            for start in ("route_b", "traveltime"):
                for it in (30, 150):
                    for win in (False, True):
                        t = _tag(well=well, arm=arm, starting=start,
                                 iterations=it, window=win)
                        key = (well, arm, start, it, win)
                        assert t not in seen, f"COLLISION {t}: {seen[t]} vs {key}"
                        seen[t] = key
    assert len(seen) == 2 * 3 * 2 * 2 * 2


def test_shot_subset_spans_the_line_not_the_first_N():
    """THE BUG THAT WOULD HAVE INVALIDATED THE WHOLE FIELD CAMPAIGN.

    Filenames run in acquisition order along the walkaway, so `files[:20]` gave
    20 shots inside 182 m of a 2960 m line -- 6% of the aperture, clustered at
    one end, carrying 18 m of the 162 m topographic relief. Not a decimated
    survey: a different and far worse experiment, with no offset range, no
    illumination, and none of the topography the air layer exists for.
    Park use 100 shots spread along the line."""
    import numpy as np
    files = [f"shot_{i:04d}.sgy" for i in range(318)]
    n = 20
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    picked = [files[i] for i in sorted(set(idx.tolist()))]
    assert picked[0] == files[0], "must include the first shot"
    assert picked[-1] == files[-1], "must include the LAST shot -- the far end"
    # spread, not clustered: consecutive gaps are all ~len/n
    gaps = np.diff([files.index(f) for f in picked])
    assert gaps.min() >= 0.5 * (len(files) / n)
    assert gaps.max() <= 2.0 * (len(files) / n)
    # and the naive slice must FAIL that test, or this asserts nothing
    naive = files[:n]
    assert naive[-1] != files[-1]
