"""Near-surface handling: air layer, Vp bounds, anisotropic gradient smoothing.

Pins three faults found by comparing our field driver against Park et al.'s
FORGE methodology. All three corrupt the same shallow zone -- the upper ~1 km
the DAS-VSP exists to constrain -- and all three were SILENT: the model looked
fine and the error went into the shallow velocities.
"""
import numpy as np
import pytest

from inversion.near_surface import (V_AIR, VP_BOUND_FIELD, air_cells, air_mask,
                                    with_air_layer, anisotropic_span,
                                    smooth2d_anisotropic, topography_relief,
                                    describe)


def test_air_survives_the_velocity_bound():
    """THE BUG: VP_BOUND=(1500,6000) clamps air (340 m/s) to 1500, which makes
    an air layer impossible to represent at all. The fix is a lower bound that
    admits slow alluvium PLUS a water_layer_mask exempting the air from the
    clamp -- this test pins the arithmetic both rely on."""
    assert VP_BOUND_FIELD[0] <= 1000.0                # Park INV2 lower bound
    assert V_AIR < VP_BOUND_FIELD[0]                  # air is BELOW the bound...
    vp = with_air_layer(np.full((40, 30), 3000.0), n_air=5)
    mask = air_mask(40, 30, 5)
    # ...so without the mask a clamp would destroy it
    clamped = np.clip(vp, *VP_BOUND_FIELD)
    assert clamped[0, 0] == VP_BOUND_FIELD[0] != V_AIR
    # with the mask, air is restored (this is what clip_params does)
    restored = np.where(mask, vp, clamped)
    assert restored[0, 0] == V_AIR and restored[-1, 0] == 3000.0


def test_air_cells_and_layer_construction():
    assert air_cells(100.0, 10.0) == 10
    assert air_cells(95.0, 10.0) == 10                 # rounds UP, never short
    assert air_cells(0.0, 10.0) == 0 and air_cells(None, 10.0) == 0
    vp = with_air_layer(np.full((20, 8), 2500.0), n_air=3)
    assert (vp[:3] == V_AIR).all() and (vp[3:] == 2500.0).all()
    assert air_mask(20, 8, 3).sum() == 3 * 8


def test_relief_decides_whether_the_air_layer_matters():
    """Do not build an air layer without measuring the relief first: if it is
    small compared with a wavelength the flat datum is defensible and the layer
    only costs grid."""
    # at 1500 m/s and 20 Hz, lambda = 75 m so the threshold is lambda/4 = 18.75 m
    assert topography_relief(np.array([0.0, 12.0, 40.0, 7.0])) == 40.0
    small = describe(200, 300, 10.0, 100.0, 1500.0, 20.0, 10.0,
                     src_z=np.array([0.0, 10.0, 4.0]))       # 10 m < 18.75
    assert "relief small" in small, small
    big = describe(200, 300, 10.0, 100.0, 1500.0, 20.0, 10.0,
                   src_z=np.array([0.0, 40.0]))              # 40 m > 18.75
    assert "MATTERS" in big, big


# --------------------------------------------------------------------------- #
# anisotropic smoothing -- Park use 2:1 and 4:1 H:V
# --------------------------------------------------------------------------- #
def test_span_is_anisotropic_and_favours_vertical_resolution():
    """A VSP's resolution is VERTICAL. Smoothing z as hard as x discards exactly
    what the fibre measures, which is what a single isotropic span did."""
    sx, sz = anisotropic_span(v_min=1500, f_max=20.0, dx=10.0, dz=10.0,
                              aspect=4.0)
    assert sx > sz, "horizontal smoothing must exceed vertical"
    assert sx == pytest.approx(4 * sz, rel=0.35)
    assert isinstance(sx, int) and isinstance(sz, int)   # smooth2d needs ints


def test_span_is_frequency_aware_like_the_isotropic_version():
    lo = anisotropic_span(1500, 5.0, 10.0, 10.0)[1]
    hi = anisotropic_span(1500, 20.0, 10.0, 10.0)[1]
    assert lo > hi                        # low frequency smooths HARDER
    for bad in ((0, 5, 10, 10), (1500, 0, 10, 10), (1500, 5, 0, 10),
                (1500, 5, 10, 0)):
        with pytest.raises(ValueError):
            anisotropic_span(*bad)


def test_anisotropic_smoothing_really_is_anisotropic():
    """Smooth a point spike and check the result is WIDER horizontally."""
    g = np.zeros((81, 81)); g[40, 40] = 1.0
    out = smooth2d_anisotropic(g, span_x=8, span_z=2)
    assert np.isfinite(out).all()
    horiz = out[40, :].sum() and float(np.abs(out[40, :] > out.max() * 0.1).sum())
    vert = float(np.abs(out[:, 40] > out.max() * 0.1).sum())
    assert horiz > vert, f"expected wider in x: {horiz} vs {vert}"
    assert out.sum() == pytest.approx(g.sum(), rel=1e-6)   # conserves mass


def test_smoothing_preserves_shape_and_reduces_roughness():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((60, 90))
    out = smooth2d_anisotropic(g, span_x=4, span_z=1)
    assert out.shape == g.shape
    rough = lambda a, ax: np.abs(np.diff(a, axis=ax)).mean()
    assert rough(out, 1) < rough(g, 1)          # smoother along x
    assert smooth2d_anisotropic(g, 0, 0).shape == g.shape   # no-op is safe


def test_describe_surfaces_the_setup():
    """Silent near-surface handling is how these faults survived a year."""
    txt = describe(200, 300, 10.0, 100.0, 1500.0, 20.0, 10.0)
    assert "air 10 rows" in txt and "1000-6000" in txt and "H:V 4:1" in txt
    assert "NO air layer" in describe(200, 300, 10.0, 0.0, 1500.0, 20.0, 10.0)
