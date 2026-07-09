"""Tests for das/geometry.py (T2).

conftest.py already puts the repo root and ADFWI on sys.path.
All correctness checks in float64.
"""

import numpy as np
import torch

from das.geometry import FiberGeometry, gauge_points


# --------------------------------------------------------------------------- #
# gauge_points convention (verified against pyber/gauge_length.py)
# --------------------------------------------------------------------------- #

def test_gauge_points_single():
    # N_g = 1 -> single center point, unit weight (pyber single_point()).
    offs, w = gauge_points(1)
    assert offs.dtype == np.float64 and w.dtype == np.float64
    assert offs.shape == (1,) and w.shape == (1,)
    assert offs[0] == 0.0
    assert w[0] == 1.0


def test_gauge_points_two():
    # N_g = 2 -> offsets +-1/2, weights 1/2, 1/2 (pyber constant().normalize()).
    offs, w = gauge_points(2)
    assert np.allclose(offs, [-0.5, 0.5])
    assert np.allclose(w, [0.5, 0.5])


def test_gauge_points_weights_sum_to_one():
    for N_g in range(1, 12):
        _, w = gauge_points(N_g)
        assert abs(w.sum() - 1.0) < 1e-15, N_g


def test_gauge_points_offsets_match_inclusive_linspace():
    for N_g in range(2, 12):
        offs, _ = gauge_points(N_g)
        assert np.allclose(offs, np.linspace(-0.5, 0.5, N_g))
        # inclusive of both endpoints
        assert offs[0] == -0.5 and offs[-1] == 0.5


def test_gauge_points_invalid():
    import pytest
    with pytest.raises(ValueError):
        gauge_points(0)


# --------------------------------------------------------------------------- #
# FiberGeometry: aligned vertical case (dz = l/2, synthetic mode)
# --------------------------------------------------------------------------- #

def _aligned_geom(**kw):
    # Channels exactly on 5 m nodes, l = 10 m = 2*dz -> aligned/dedup-friendly.
    params = dict(x_well=250.0, z_top=100.0, n_channels=20,
                  dch=1.02, l=10.0, dx=5.0, dz=5.0, snap_to_nodes=False)
    params.update(kw)
    return FiberGeometry(**params)


def test_unit_tangent_norm():
    g = _aligned_geom()
    norm = np.sqrt(g.e_x ** 2 + g.e_z ** 2)
    assert np.allclose(norm, 1.0)
    # vertical fiber -> e = (0, 1)
    assert np.allclose(g.e_x, 0.0)
    assert np.allclose(g.e_z, 1.0)


def test_endpoint_separation_equals_l():
    g = _aligned_geom()
    for c in range(g.C):
        z_minus = g.endpoint_z(c, -1)
        z_plus = g.endpoint_z(c, +1)
        assert abs((z_plus - z_minus) - g.l) < 1e-12


def test_dedup_aligned_vertical():
    g = _aligned_geom()
    C = g.C
    # Aligned vertical fiber -> n_rcv == C + 2.
    assert g.n_rcv == C + 2
    # idx_plus - idx_minus == 2 everywhere, idx_minus[c] == c.
    diff = (g.idx_plus - g.idx_minus).numpy()
    assert np.all(diff == 2)
    assert np.array_equal(g.idx_minus.numpy(), np.arange(C))
    assert np.array_equal(g.idx_plus.numpy(), np.arange(C) + 2)


def test_index_maps_are_longtensors():
    g = _aligned_geom()
    assert g.idx_minus.dtype == torch.long
    assert g.idx_plus.dtype == torch.long
    assert g.idx_minus.shape == (g.C,)
    assert g.idx_plus.shape == (g.C,)


def test_rcv_pos_unique_and_indexable():
    g = _aligned_geom()
    # all receiver positions unique
    assert len(set(g.rcv_pos)) == len(g.rcv_pos)
    # every index map entry is a valid rcv_pos index
    for c in range(g.C):
        assert 0 <= int(g.idx_minus[c]) < g.n_rcv
        assert 0 <= int(g.idx_plus[c]) < g.n_rcv


def test_channel_count_synthetic():
    g = _aligned_geom(n_channels=20)
    assert g.C == 20
    assert g.channel_z.shape == (20,)
    # exactly on nodes with spacing dz
    assert np.allclose(np.diff(g.channel_z), g.dz)


def test_float64_dtypes():
    g = _aligned_geom()
    assert g.channel_z.dtype == np.float64
    assert g.e_x.dtype == np.float64
    assert g.e_z.dtype == np.float64


# --------------------------------------------------------------------------- #
# FiberGeometry: field mode (snap_to_nodes=True)
# --------------------------------------------------------------------------- #

def test_snapping_error_within_half_dch():
    # Real FORGE-like numbers: dch = 1.02, dz = 5.0. Nearest channel per node.
    g = FiberGeometry(x_well=250.0, z_top=100.0, n_channels=1010,
                      dch=1.02, l=10.0, dx=5.0, dz=5.0, snap_to_nodes=True)
    # each selected channel must be within dch/2 of its grid node
    node_z = np.round(g.channel_z / g.dz) * g.dz
    err = np.abs(g.channel_z - node_z)
    assert err.max() <= g.dch / 2.0 + 1e-12


def test_field_mode_channel_count_matches_span():
    # Span [z_top, z_top + (n-1)*dch]; number of channels = number of grid nodes
    # in that span.
    z_top, n, dch, dz = 100.0, 1010.0, 1.02, 5.0
    z_top = 100.0
    g = FiberGeometry(x_well=250.0, z_top=z_top, n_channels=int(n),
                      dch=dch, l=10.0, dx=5.0, dz=dz, snap_to_nodes=True)
    z_bottom = z_top + (int(n) - 1) * dch
    k_lo = int(np.ceil(z_top / dz))
    k_hi = int(np.floor(z_bottom / dz))
    expected = k_hi - k_lo + 1
    assert g.C == expected


def test_field_mode_endpoint_separation_and_norm():
    g = FiberGeometry(x_well=250.0, z_top=100.0, n_channels=1010,
                      dch=1.02, l=10.0, dx=5.0, dz=5.0, snap_to_nodes=True)
    assert np.allclose(np.sqrt(g.e_x ** 2 + g.e_z ** 2), 1.0)
    for c in range(g.C):
        assert abs((g.endpoint_z(c, +1) - g.endpoint_z(c, -1)) - g.l) < 1e-12


def test_field_mode_dedup_aligned():
    # z_top on a node, dch snaps every node to a channel, dz = l/2 -> C + 2.
    g = FiberGeometry(x_well=250.0, z_top=100.0, n_channels=1010,
                      dch=1.02, l=10.0, dx=5.0, dz=5.0, snap_to_nodes=True)
    assert g.n_rcv == g.C + 2
    assert np.all((g.idx_plus - g.idx_minus).numpy() == 2)
