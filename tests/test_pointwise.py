"""Tests for the E5 pointwise operator (deviated / curved fibers).

Validation strategy (E5 has no closed form except its definition, so we pin it
against cases where the answer IS known):

1. Linear velocity field -> exact. Central-difference derivatives and trapezoid
   quadrature are both exact for a field linear in space, so E5 returns the
   analytic axial strain rate ``e^T eps_dot e`` to machine precision, for a
   vertical AND a tilted fiber (this validates the tangent contraction).
2. Convergence to E3 on a vertical fiber with a smooth plane wave: E5 -> E3 as
   N_g grows / the grid refines (E3 is the exact straight-fiber operator).
3. THE curvature gate: on a curved fiber, the identity
   d(v.e)/ds = e^T eps_dot e + v.(de/ds) must hold, i.e. E5 (which returns
   e^T eps_dot e) differs from the naive along-fiber difference of v_e by
   exactly the curvature term v.(de/ds) -- proving E5 excludes it.
4. Linearity and gradient flow (differentiability).
"""

import numpy as np
import pytest
import torch

from das.geometry import FiberGeometry
from das.das_layer import DASObservationLayer
from das.geometry_pointwise import PointwiseFiberGeometry
from das.das_layer_pointwise import DASPointwiseLayer

DX = DZ = 5.0


def records(geom, vx_fn, vz_fn, T=1):
    """Sample (vx, vz) at the geometry's rcv_pos nodes -> [1, T, R] each."""
    xs = np.array([ix * geom.dx for (_iz, ix) in geom.rcv_pos])
    zs = np.array([iz * geom.dz for (iz, _ix) in geom.rcv_pos])
    u = np.stack([vx_fn(xs, zs, t) for t in range(T)])[None]   # [1, T, R]
    w = np.stack([vz_fn(xs, zs, t) for t in range(T)])[None]
    return torch.from_numpy(u.copy()), torch.from_numpy(w.copy())


# --------------------------------------------------------------------------- #
# 1. linear field -> exact axial strain rate
# --------------------------------------------------------------------------- #

def test_linear_field_vertical_exact():
    g = PointwiseFiberGeometry.vertical(x_well=250.0, z_top=100.0, z_bot=300.0,
                                        n_channels=20, dch=5.0, l=10.0,
                                        dx=DX, dz=DZ, N_g=5)
    layer = DASPointwiseLayer(g)
    a = 0.0013                                   # vz = a*z -> dvz/dz = a
    u, w = records(g, lambda x, z, t: np.zeros_like(x),
                   lambda x, z, t: a * z)
    out = layer(u, w).numpy()[0, 0]              # [C]
    assert np.allclose(out, a, atol=1e-12), f"vertical axial != a: {out[:3]}"


def test_linear_field_tilted_exact():
    # tilted straight fiber, general linear field: vx = A x + B z, vz = C x + D z
    A, B, C, D = 7e-4, -3e-4, 5e-4, 9e-4
    dip = 35.0
    g = PointwiseFiberGeometry.tilted(x_top=300.0, z_top=120.0, dip_deg=dip,
                                      length=200.0, n_channels=15, dch=5.0,
                                      l=10.0, dx=DX, dz=DZ, N_g=5)
    layer = DASPointwiseLayer(g)
    u, w = records(g, lambda x, z, t: A * x + B * z,
                   lambda x, z, t: C * x + D * z)
    out = layer(u, w).numpy()[0, 0]
    th = np.radians(dip)
    ex, ez = np.sin(th), np.cos(th)
    analytic = ex * ex * A + ex * ez * (B + C) + ez * ez * D
    assert np.allclose(out, analytic, atol=1e-12), (
        f"tilted axial {out[0]:.6e} != analytic {analytic:.6e}")


# --------------------------------------------------------------------------- #
# 2. reduces to E3 on a vertical fiber, EXACTLY, for a quadratic field
# --------------------------------------------------------------------------- #

def test_reduces_to_e3_quadratic():
    # For v_z quadratic in z, dv_z/dz is linear: central differencing is exact
    # and the trapezoid rule is exact, so E5 must equal the E3 endpoint
    # difference to machine precision. (For a general smooth field the two
    # operators differ at grid scale -- they share only the continuum limit.)
    alpha, beta, gamma = 3e-6, 7e-4, 0.2

    ge3 = FiberGeometry(x_well=250.0, z_top=100.0, n_channels=20,
                        l=10.0, dx=DX, dz=DZ, snap_to_nodes=False)
    layer3 = DASObservationLayer(ge3)
    zr = np.array([iz * DZ for (iz, _ix) in ge3.rcv_pos], float)
    vz3 = (alpha * zr ** 2 + beta * zr + gamma)[None, None]        # [1,1,R]
    out3 = layer3(torch.zeros_like(torch.from_numpy(vz3)),
                  torch.from_numpy(vz3.copy())).numpy()[0, 0]      # [C]

    gp = PointwiseFiberGeometry.vertical(250.0, 100.0, 100.0 + 19 * DZ,
                                         20, DZ, 10.0, dx=DX, dz=DZ, N_g=5)
    layer5 = DASPointwiseLayer(gp)
    u5, w5 = records(gp, lambda x, z, t: np.zeros_like(x),
                     lambda x, z, t: alpha * z ** 2 + beta * z + gamma)
    out5 = layer5(u5, w5).numpy()[0, 0]

    assert np.allclose(out5, out3, atol=1e-9), (
        f"E5 != E3 on quadratic field: max diff {np.abs(out5-out3).max():.2e}")


# --------------------------------------------------------------------------- #
# 3. curvature gate: E5 returns e^T eps_dot e, NOT the along-fiber difference
# --------------------------------------------------------------------------- #

def test_curvature_excluded():
    # a gently curved fiber (quarter-ish arc), linear velocity field so the
    # pointwise strain rate is exact; verify
    #   d(v.e)/ds  ==  E5(e^T eps_dot e)  +  v.(de/ds)
    # and that the curvature term is NON-negligible (so E5 != naive diff).
    R = 400.0                                    # arc radius [m]
    th = np.linspace(0.0, np.pi / 3, 60)         # 0..60 deg
    verts = np.stack([300.0 + R * (1 - np.cos(th)), 150.0 + R * np.sin(th)], 1)
    g = PointwiseFiberGeometry(verts, n_channels=40, dch=5.0, l=10.0,
                               dx=DX, dz=DZ, N_g=1)     # N_g=1: pointwise centre
    layer = DASPointwiseLayer(g)

    A, B, C, D = 6e-4, 2e-4, -4e-4, 8e-4
    u, w = records(g, lambda x, z, t: A * x + B * z,
                   lambda x, z, t: C * x + D * z)
    e5 = layer(u, w).numpy()[0, 0]               # [C] = e^T eps_dot e per channel

    ex, ez = g.e_x, g.e_z
    vx_c = A * g.channel_x + B * g.channel_z
    vz_c = C * g.channel_x + D * g.channel_z
    v_e = ex * vx_c + ez * vz_c                   # v.e at channel centres
    dch = g.dch

    # interior channels: central differences along the fibre
    c = np.arange(1, g.C - 1)
    along = (v_e[c + 1] - v_e[c - 1]) / (2 * dch)              # d(v.e)/ds
    de_ds_x = (ex[c + 1] - ex[c - 1]) / (2 * dch)
    de_ds_z = (ez[c + 1] - ez[c - 1]) / (2 * dch)
    curv = vx_c[c] * de_ds_x + vz_c[c] * de_ds_z              # v.(de/ds)

    residual = np.abs(along - (e5[c] + curv))
    # along-fibre difference is 2nd-order: residual ~ O(dch^2), small
    assert residual.max() < 1e-4, f"identity violated: {residual.max():.2e}"
    # and the curvature term is genuinely present (E5 != naive along-fibre diff)
    assert np.abs(curv).max() > 1e-5, "curvature term negligible - weak test"
    assert np.abs(along - e5[c]).max() > 10 * residual.max(), (
        "E5 indistinguishable from the naive along-fibre difference")


# --------------------------------------------------------------------------- #
# 4. linearity + gradient flow
# --------------------------------------------------------------------------- #

def test_linearity_and_grad():
    g = PointwiseFiberGeometry.tilted(300.0, 120.0, 20.0, 150.0, 12, 5.0, 10.0,
                                      dx=DX, dz=DZ, N_g=5)
    layer = DASPointwiseLayer(g)
    gen = torch.Generator().manual_seed(0)
    u1 = torch.randn(1, 8, g.n_rcv, dtype=torch.float64, generator=gen)
    w1 = torch.randn(1, 8, g.n_rcv, dtype=torch.float64, generator=gen)
    u2 = torch.randn(1, 8, g.n_rcv, dtype=torch.float64, generator=gen)
    w2 = torch.randn(1, 8, g.n_rcv, dtype=torch.float64, generator=gen)
    a, b = 2.0, -1.5
    lhs = layer(a * u1 + b * u2, a * w1 + b * w2)
    rhs = a * layer(u1, w1) + b * layer(u2, w2)
    assert torch.allclose(lhs, rhs, atol=1e-12)

    u = torch.randn(1, 8, g.n_rcv, dtype=torch.float64, requires_grad=True,
                    generator=None)
    w = torch.randn(1, 8, g.n_rcv, dtype=torch.float64, requires_grad=True)
    layer(u, w).pow(2).sum().backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(w.grad).all()
    assert u.grad.abs().max() > 0 and w.grad.abs().max() > 0


def test_strain_option():
    g = PointwiseFiberGeometry.vertical(250.0, 100.0, 200.0, 15, 5.0, 10.0,
                                        dx=DX, dz=DZ, N_g=5)
    rate = DASPointwiseLayer(g, output="strain_rate")
    strain = DASPointwiseLayer(g, output="strain", dt=4e-4)
    gen = torch.Generator().manual_seed(1)
    u = torch.randn(1, 20, g.n_rcv, dtype=torch.float64, generator=gen)
    w = torch.randn(1, 20, g.n_rcv, dtype=torch.float64, generator=gen)
    assert torch.equal(strain(u, w), torch.cumsum(rate(u, w), dim=1) * 4e-4)
    with pytest.raises(ValueError):
        DASPointwiseLayer(g, output="strain")     # missing dt
