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
    assert "flat datum ok" in small, small
    big = describe(200, 300, 10.0, 100.0, 1500.0, 20.0, 10.0,
                   src_z=np.array([0.0, 40.0]))              # 40 m > 18.75
    assert "AIR LAYER REQUIRED" in big, big


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


# --------------------------------------------------------------------------- #
# topography-following air layer -- MEASURED at FORGE, not assumed
# --------------------------------------------------------------------------- #
from inversion.near_surface import (surface_profile, air_mask_topo,      # noqa: E402
                                    with_air_layer_topo)

# the real FORGE ramp: 161.6 m over ~2960 m, corr(x,z) = +0.994, 318 shots
FORGE_SX = np.array([-1547.0, -691.0, 214.0, 1412.0])
FORGE_SZ = np.array([0.0, 65.0, 108.2, 161.6])


def test_a_uniform_air_slab_is_wrong_at_forge():
    """THE POINT: at one end of the line the ground IS the datum (zero air), at
    the other it is 162 m below. A constant-thickness slab cannot represent
    that, which is what "the inclined nature of the surface topography" means."""
    gd = surface_profile(FORGE_SX, FORGE_SZ, nx=60, dx=50.0, x0=-1547.0)
    assert gd[0] == pytest.approx(0.0, abs=1e-9)      # datum end: NO air
    assert gd[-1] > 130.0                             # far end: lots of air
    assert np.all(np.diff(gd) >= -1e-9), "the measured profile is monotonic"


def test_air_mask_follows_the_ramp():
    gd = surface_profile(FORGE_SX, FORGE_SZ, nx=60, dx=50.0, x0=-1547.0)
    m = air_mask_topo(40, 60, gd, dz=10.0)
    per_col = m.sum(axis=0)
    assert per_col[0] == 0                            # no air at the datum end
    assert per_col[-1] >= 13                          # ~162 m / 10 m
    assert np.all(np.diff(per_col) >= 0), "air thickness must track the ramp"


def test_topographic_air_layer_sets_only_the_air():
    gd = surface_profile(FORGE_SX, FORGE_SZ, nx=40, dx=80.0, x0=-1547.0)
    vp = with_air_layer_topo(np.full((30, 40), 3000.0), gd, dz=10.0)
    m = air_mask_topo(30, 40, gd, dz=10.0)
    assert (vp[m] == V_AIR).all() and (vp[~m] == 3000.0).all()
    # and the ROCK column under the datum end is untouched top to bottom
    assert (vp[:, 0] == 3000.0).all()


def test_describe_quantifies_the_flat_datum_error():
    """A flat datum fabricates a free-surface ghost 2h/v late. At FORGE that is
    215 ms = 8.6 half-cycles at 20 Hz -- far past cycle skipping, an invented
    arrival. The log must say so rather than silently proceeding."""
    txt = describe(200, 300, 10.0, 162.0, 1500.0, 20.0, 10.0,
                   src_z=FORGE_SZ)
    assert "AIR LAYER REQUIRED" in txt
    assert "half-cycles" in txt and "ghost" in txt
