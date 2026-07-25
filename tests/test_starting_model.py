"""Route B starting models: data independence, Vs seeding, physics guards."""
import numpy as np
import pytest

from inversion.starting_model import (SQRT3, linear_vz, vs_from_vp,
                                      clip_to_bounds, smooth_model,
                                      poisson_clamp)


def test_linear_vz_is_data_independent_and_1d():
    v = linear_vz(88, 200, 1500.0, 4000.0)
    assert v.shape == (88, 200)
    assert np.allclose(v[:, 0], v[:, -1])            # laterally constant -> 1-D
    assert v[0, 0] == pytest.approx(1500.0)
    assert v[-1, 0] == pytest.approx(4000.0)
    assert np.all(np.diff(v[:, 0]) > 0)              # monotonic with depth


def test_linear_vz_water_layer_pinned():
    v = linear_vz(88, 200, 1500.0, 4000.0, water_rows=12, v_water=1500.0)
    assert np.allclose(v[:12], 1500.0)
    assert v[12, 0] > 1500.0


def test_vs_seed_is_poisson_solid():
    """sqrt(3) <=> nu = 0.25 (lambda = mu): the universal isotropic default,
    NOT a basin-calibrated relation like Castagna."""
    assert SQRT3 == pytest.approx(1.7320508, abs=1e-6)
    vp = np.full((10, 10), 5200.0)
    vs = vs_from_vp(vp)
    assert vs[0, 0] == pytest.approx(5200.0 / 1.7320508, rel=1e-6)
    # Poisson ratio implied by Vp/Vs = sqrt(3) is exactly 0.25
    r = vp / vs
    nu = (r ** 2 - 2) / (2 * (r ** 2 - 1))
    assert np.allclose(nu, 0.25)


def test_vs_seed_depth_graded_ratio():
    """FORGE-style: a higher Vp/Vs in the sedimentary cover, ~1.73 in basement."""
    vp = np.full((100, 5), 4000.0)
    vs = vs_from_vp(vp, ratio=2.2, depth_ratio=[(0.0, 2.2), (1000.0, 1.73)], dz=20.0)
    assert vs[0, 0] == pytest.approx(4000.0 / 2.2)       # cover
    assert vs[-1, 0] == pytest.approx(4000.0 / 1.73)     # basement (z >= 1000 m)
    assert vs[49, 0] == pytest.approx(4000.0 / 2.2)      # z = 980 m, still cover
    assert vs[50, 0] == pytest.approx(4000.0 / 1.73)     # z = 1000 m, basement


def test_vs_seed_depth_ratio_needs_dz():
    with pytest.raises(ValueError):
        vs_from_vp(np.ones((4, 4)) * 3000, depth_ratio=[(0.0, 2.0)])


def test_poisson_clamp_enforces_stability():
    """vs <= vp/1.5; below sqrt(2) the elastic scheme diverges."""
    vp = np.full((5, 5), 3000.0)
    vs = np.full((5, 5), 2500.0)                     # illegal: vp/vs = 1.2
    out = poisson_clamp(vp, vs, min_vp_vs=1.5)
    assert np.all(out <= vp / 1.5 + 1e-9)
    assert np.all(vp / out >= 1.5 - 1e-9)
    # a legal model is untouched
    ok = np.full((5, 5), 1500.0)
    assert np.allclose(poisson_clamp(vp, ok), ok)


def test_smoothing_and_bounds():
    rng = np.random.default_rng(0)
    v = 3000 + 500 * rng.standard_normal((40, 60))
    s = smooth_model(v, 4.0)
    assert s.std() < v.std()                          # smoother
    assert s.mean() == pytest.approx(v.mean(), rel=0.05)
    c = clip_to_bounds(v, 2000, 4000)
    assert c.min() >= 2000 and c.max() <= 4000
