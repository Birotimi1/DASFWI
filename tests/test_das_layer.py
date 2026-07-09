"""Tests for das/das_layer.py (T3).

conftest.py puts the repo root on sys.path. Correctness thresholds in float64.
"""

import numpy as np
import pytest
import torch

from das.geometry import FiberGeometry
from das.das_layer import DASObservationLayer


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def make_geom(n_channels=20):
    # Synthetic mode: channels exactly on 5 m nodes, l = 10 m = 2*dz, so the
    # gauge endpoints z_c +- l/2 are exactly the endpoint-receiver depths.
    return FiberGeometry(x_well=250.0, z_top=100.0, n_channels=n_channels,
                         dch=1.02, l=10.0, dx=5.0, dz=5.0, snap_to_nodes=False)


def rcv_depths(geom):
    """Physical depth of each receiver in rcv_pos order."""
    return np.array([k_z * geom.dz for (k_z, _k_x) in geom.rcv_pos],
                    dtype=np.float64)


def gaussian(t, t0=0.35, sigma=0.05):
    return np.exp(-((t - t0) / sigma) ** 2)


# --------------------------------------------------------------------------- #
# construction / validation
# --------------------------------------------------------------------------- #

def test_invalid_output_raises():
    with pytest.raises(ValueError):
        DASObservationLayer(make_geom(), output="velocity")


def test_strain_without_dt_raises():
    with pytest.raises(ValueError):
        DASObservationLayer(make_geom(), output="strain")  # dt missing


def test_buffers_registered():
    layer = DASObservationLayer(make_geom())
    buffers = dict(layer.named_buffers())
    assert set(buffers) == {"idx_plus", "idx_minus", "e_x", "e_z"}
    assert buffers["idx_plus"].dtype == torch.long
    assert buffers["idx_minus"].dtype == torch.long
    assert buffers["e_x"].dtype == torch.float64
    assert buffers["e_z"].dtype == torch.float64


# --------------------------------------------------------------------------- #
# shape / dtype / device
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_shape_dtype_device(dtype):
    geom = make_geom()
    layer = DASObservationLayer(geom)
    S, T, R, C = 3, 40, geom.n_rcv, geom.C
    rcv_u = torch.randn(S, T, R, dtype=dtype)
    rcv_w = torch.randn(S, T, R, dtype=dtype)
    out = layer(rcv_u, rcv_w)
    assert out.shape == (S, T, C)
    assert out.dtype == dtype
    assert out.device == rcv_u.device


# --------------------------------------------------------------------------- #
# analytic identity (E6-style): plane wave v_e(z, t) = f(t - z/c0)
# --------------------------------------------------------------------------- #

def test_analytic_identity_float64():
    geom = make_geom(n_channels=20)
    layer = DASObservationLayer(geom)

    c0 = 2000.0
    nt = 600
    dt = 5e-4
    t = np.arange(nt) * dt                      # [T]
    z_r = rcv_depths(geom)                      # [R]

    # v_e sampled at the receivers: vertical fiber -> v_e = w, u = 0.
    v_e = gaussian(t[:, None] - z_r[None, :] / c0)      # [T, R]
    rcv_w = torch.from_numpy(v_e[None, ...].copy())     # [1, T, R] float64
    rcv_u = torch.zeros_like(rcv_w)

    out = layer(rcv_u, rcv_w).numpy()[0]                # [T, C]

    z_c = geom.channel_z                                # [C]
    expected = (gaussian(t[:, None] - (z_c[None, :] + geom.l / 2) / c0)
                - gaussian(t[:, None] - (z_c[None, :] - geom.l / 2) / c0)
                ) / geom.l

    num = np.abs(out - expected).max()
    den = np.abs(expected).max()
    assert num / den <= 1e-12


# --------------------------------------------------------------------------- #
# strain option (E4)
# --------------------------------------------------------------------------- #

def test_strain_equals_cumsum_dt():
    geom = make_geom()
    dt = 4e-4
    layer_rate = DASObservationLayer(geom, output="strain_rate")
    layer_strain = DASObservationLayer(geom, output="strain", dt=dt)

    S, T, R = 2, 50, geom.n_rcv
    g = torch.Generator().manual_seed(0)
    rcv_u = torch.randn(S, T, R, dtype=torch.float64, generator=g)
    rcv_w = torch.randn(S, T, R, dtype=torch.float64, generator=g)

    rate = layer_rate(rcv_u, rcv_w)
    strain = layer_strain(rcv_u, rcv_w)
    assert torch.equal(strain, torch.cumsum(rate, dim=1) * dt)


# --------------------------------------------------------------------------- #
# linearity
# --------------------------------------------------------------------------- #

def test_linearity():
    geom = make_geom()
    layer = DASObservationLayer(geom)
    S, T, R = 2, 30, geom.n_rcv
    g = torch.Generator().manual_seed(1)
    u1 = torch.randn(S, T, R, dtype=torch.float64, generator=g)
    w1 = torch.randn(S, T, R, dtype=torch.float64, generator=g)
    u2 = torch.randn(S, T, R, dtype=torch.float64, generator=g)
    w2 = torch.randn(S, T, R, dtype=torch.float64, generator=g)
    a, b = 2.5, -1.25  # exactly representable

    lhs = layer(a * u1 + b * u2, a * w1 + b * w2)
    rhs = a * layer(u1, w1) + b * layer(u2, w2)
    assert torch.allclose(lhs, rhs, rtol=0.0, atol=1e-15)


# --------------------------------------------------------------------------- #
# channel_mask
# --------------------------------------------------------------------------- #

def test_channel_mask_logic():
    geom = make_geom(n_channels=5)
    layer = DASObservationLayer(geom)
    C, R = geom.C, geom.n_rcv  # R == C + 2 (aligned)
    idx_m = geom.idx_minus.numpy()
    idx_p = geom.idx_plus.numpy()

    # shot 0: all receivers alive -> all channels alive
    # shot 1: kill channel 2's minus endpoint only
    # shot 2: kill channel 2's plus endpoint only
    # shot 3: kill both endpoints of channel 2
    mask = torch.ones(4, R, dtype=torch.bool)
    mask[1, idx_m[2]] = False
    mask[2, idx_p[2]] = False
    mask[3, idx_m[2]] = False
    mask[3, idx_p[2]] = False

    cm = layer.channel_mask(mask)
    assert cm.shape == (4, C)
    assert cm[0].all()

    # channel dies iff EITHER endpoint is masked
    for s in (1, 2, 3):
        assert not cm[s, 2]

    # killing channel 2's endpoints also kills the channels that SHARE them:
    # idx_m[2] is idx_p[0] (shared node), idx_p[2] is idx_m[4].
    assert not cm[1, 0]      # shares receiver idx_m[2] == idx_p[0]
    assert not cm[2, 4]      # shares receiver idx_p[2] == idx_m[4]
    # channels 1 and 3 keep both endpoints in shots 1 and 2
    assert cm[1, 1] and cm[1, 3]
    assert cm[2, 1] and cm[2, 3]


# --------------------------------------------------------------------------- #
# gradient flow (autograd through the layer)
# --------------------------------------------------------------------------- #

def test_gradient_flows():
    geom = make_geom()
    layer = DASObservationLayer(geom)
    S, T, R = 1, 20, geom.n_rcv
    rcv_u = torch.randn(S, T, R, dtype=torch.float64, requires_grad=True)
    rcv_w = torch.randn(S, T, R, dtype=torch.float64, requires_grad=True)

    out = layer(rcv_u, rcv_w)
    out.sum().backward()

    assert rcv_w.grad is not None
    assert torch.isfinite(rcv_w.grad).all()
    # Every receiver is an endpoint of some channel in the aligned case, so the
    # vertical-velocity gradient must be nonzero somewhere...
    assert rcv_w.grad.abs().max() > 0
    # ...specifically at the extreme receivers, which each serve ONE channel
    # (contribution -+ 1/l), while interior receivers serve two channels with
    # canceling +-1/l contributions under a sum() loss.
    assert torch.allclose(rcv_w.grad[..., 0],
                          torch.full_like(rcv_w.grad[..., 0], -1.0 / geom.l))
    assert torch.allclose(rcv_w.grad[..., -1],
                          torch.full_like(rcv_w.grad[..., -1], 1.0 / geom.l))
    # Vertical fiber: e_x = 0 -> gradient w.r.t. horizontal records is zero
    # but still defined (graph is connected through e_x * u).
    assert rcv_u.grad is not None
    assert torch.all(rcv_u.grad == 0)


def test_gradient_flows_strain():
    geom = make_geom()
    layer = DASObservationLayer(geom, output="strain", dt=4e-4)
    rcv_u = torch.randn(1, 15, geom.n_rcv, dtype=torch.float64,
                        requires_grad=True)
    rcv_w = torch.randn(1, 15, geom.n_rcv, dtype=torch.float64,
                        requires_grad=True)
    layer(rcv_u, rcv_w).pow(2).sum().backward()
    assert rcv_w.grad is not None and torch.isfinite(rcv_w.grad).all()
    assert rcv_w.grad.abs().max() > 0


# --------------------------------------------------------------------------- #
# pointwise stub (E5)
# --------------------------------------------------------------------------- #

def test_forward_pointwise_not_implemented():
    layer = DASObservationLayer(make_geom())
    with pytest.raises(NotImplementedError):
        layer.forward_pointwise(None, None)
    # docstring must carry the E5 formula and the curvature warning
    doc = layer.forward_pointwise.__doc__
    assert "E5" in doc
    assert "de/ds" in doc
    assert "curvature" in doc.lower()
