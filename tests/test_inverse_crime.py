"""Tests for T7: FORGE proxy model + miniature inverse-crime inversion.

The miniature run is the spec's local end-to-end check (101x101, 4 shots,
20 iterations): loss must decrease monotone-ish and the velocity update must
have the right sign in the right region. Full-scale runs are HPC work.
"""

import numpy as np
import pytest
import torch

from forge.proxy_model import (forge_proxy_vp, gardner_rho, forge_fibers,
                               V_AIR, V_I, V_II, V_III, Z_AIR, Z_I_BOTTOM,
                               Z_II_BOTTOM, DZ)
from inversion.run_inverse_crime import run_inverse_crime


# --------------------------------------------------------------------------- #
# proxy model zones (E11)
# --------------------------------------------------------------------------- #

def test_proxy_zones():
    nz, nx = 260, 40                      # 1.3 km deep at 5 m
    vp = forge_proxy_vp(nz, nx)
    assert vp.shape == (nz, nx) and vp.dtype == np.float64
    z = np.arange(nz) * DZ

    air = z < Z_AIR
    assert np.all(vp[air] == V_AIR)

    zone1 = (z >= Z_AIR) & (z < Z_I_BOTTOM)
    assert vp[zone1].min() >= V_I[0] - 1e-9
    assert vp[zone1].max() <= V_I[1] + 1e-9

    zone2 = (z >= Z_I_BOTTOM) & (z < Z_II_BOTTOM)
    assert vp[zone2].min() >= V_II[0] - 1e-9
    assert vp[zone2].max() <= V_II[1] + 1e-9

    zone3 = z >= Z_II_BOTTOM
    assert vp[zone3].min() >= V_III[0] - 1e-9
    assert vp[zone3].max() <= V_III[1] + 1e-9

    # laterally homogeneous; non-decreasing with depth below the air layer
    assert np.allclose(vp, vp[:, :1])
    below_air = vp[~air, 0]
    assert np.all(np.diff(below_air) >= -1e-9)


def test_gardner_rho_air():
    vp = forge_proxy_vp(120, 10)
    rho = gardner_rho(vp)
    n_air = int(round(Z_AIR / DZ))
    assert np.all(rho[:n_air] == 1.225)
    assert np.all(rho[n_air:] > 1000.0)   # Gardner for rocks


def test_forge_fibers_two_wells():
    ga, gb = forge_fibers(nz=260, x_well_a=1000.0, x_well_b=1400.0)
    assert ga.C == gb.C > 0
    xa = {kx for (_kz, kx) in ga.rcv_pos}
    xb = {kx for (_kz, kx) in gb.rcv_pos}
    assert xa == {200} and xb == {280}    # x_well / dx
    assert ga.n_rcv == ga.C + 2 and gb.n_rcv == gb.C + 2


# --------------------------------------------------------------------------- #
# miniature end-to-end inverse crime (E12 wiring; spec T7 local check)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_miniature_inversion():
    nz = nx = 101
    zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    bump = 150.0 * np.exp(-(((zz - 30) ** 2 + (xx - 40) ** 2) / (2 * 8.0 ** 2)))
    vp_true = np.full((nz, nx), 2000.0) + bump
    vp_init = np.full((nz, nx), 2000.0)

    result = run_inverse_crime(dict(vp_true=vp_true, vp_init=vp_init))
    losses = np.asarray(result["iter_loss"])

    # 4 shots, 20 iterations ran
    assert len(losses) == 20
    assert np.isfinite(losses).all()

    # monotone-ish decrease: clearly lower at the end, most steps decreasing
    assert losses[-1] < losses[0], f"no overall decrease: {losses}"
    frac_down = np.mean(np.diff(losses) < 0)
    assert frac_down >= 0.6, (
        f"only {frac_down:.0%} of steps decreased: {losses}")

    # velocity update: right sign (positive, vp_true > vp_init) in the right
    # region (inside the bump), and clearly stronger inside than outside
    delta = result["delta_vp"]
    inside = bump > 50.0
    outside = bump < 5.0
    assert delta[inside].mean() > 0, "update has the wrong sign in the bump"
    assert delta[inside].mean() > 5.0 * abs(delta[outside].mean()), (
        "update not concentrated in the true-anomaly region")
    assert np.abs(delta).max() > 1.0, "no visible velocity update (m/s)"
