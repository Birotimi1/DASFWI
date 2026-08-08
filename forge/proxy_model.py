"""FORGE proxy velocity model and inverse-crime data generation (T7, E11-E12).

Proxy vp on the 5 m grid (spec E11; depths are MODEL depths, i.e. they include
the ~0.1 km air offset above the ground surface):

    air      :      0 - z_air (~0.10 km) : v_air (default 340 m/s)
    zone I   :  z_air - ~0.45 km         : ~1.5 -> 2.5 km/s (linear)
    zone II  :  ~0.45 - ~1.10 km         : ~2.5 -> 4.5 km/s (linear)
    zone III :  below ~1.10 km           : granitoid ~5.5 -> 5.9 km/s (linear)

The air layer is a low-velocity slab at the top of the grid; sources and
receivers must be placed at or below the air-ground interface. Exact air
handling (velocity/density values, free surface flag) is parameterized and can
be revisited at HPC stage without touching the rest of the chain.

Two vertical DAS fibers stand at the (parameterized) well x-positions of
78A-32 and 78B-32; a surface vibroseis line provides Ricker sources
w(t) = (1 - 2 pi^2 f0^2 (t-t0)^2) exp(-pi^2 f0^2 (t-t0)^2)  (this is exactly
ADFWI's utils.wavelets.wavelet(..., type='Ricker')).

`generate_observed` runs the propagator on the TRUE model and stores
record_data({"strain_rate": DASObservationLayer(u, w)}) - inverse crime by
construction (same propagator family, same operator, no noise).

A Marmousi2 variant is provided through ADFWI's own loader
(utils/velocityDemo.py: load_marmousi_model + resample_marmousi_model); it
downloads SEGY files on first use, so it is NOT exercised by the local tests.
"""

import numpy as np
import torch

from ADFWI.model import AcousticModel
from ADFWI.survey import Source, Receiver, Survey, SeismicData
from ADFWI.propagator import AcousticPropagator
from ADFWI.utils.wavelets import wavelet

from das.geometry import FiberGeometry
from das.das_layer import DASObservationLayer

# FORGE / spec E0 constants
DX = DZ = 5.0
GAUGE_L = 10.0
DCH = 1.02

# zone boundaries and velocities (spec E11), all in meters / m/s
Z_AIR = 100.0
Z_I_BOTTOM = 450.0
Z_II_BOTTOM = 1100.0
V_AIR = 340.0
V_I = (1500.0, 2500.0)
V_II = (2500.0, 4500.0)
V_III = (5500.0, 5900.0)
V_III_BOTTOM_DEPTH = 2000.0   # depth at which zone III reaches V_III[1]


def forge_proxy_vp(nz, nx, dz=DZ, z_air=Z_AIR, dip_m_per_km=0.0, dx=None):
    """Piecewise-linear FORGE proxy vp [nz, nx] (float64), zones per E11.

    Zone III grades linearly from V_III[0] at Z_II_BOTTOM to V_III[1] at
    V_III_BOTTOM_DEPTH and stays at V_III[1] below that.

    `dip_m_per_km` TILTS the zone boundaries, shallowing them with increasing x
    (Park's sense: their basement rises from ~1.15 km on the left to ~0.5 km on
    the right, i.e. about -230 m/km).

    >>> WHY THIS EXISTS: THE DECISIVE TEST. <<<
    The profile was laterally homogeneous, so a DAS-only inversion could not be
    scored on lateral structure -- there was none to recover, and the synthetic
    could never answer whether our failure to reproduce Park's dip is a defect
    or a limit of the acquisition.
    It matters because Park do NOT invert FORGE from DAS alone: their VM1 comes
    from 3-D SURFACE SEISMIC tomography (INV1) and the DAS-VSP FWI (INV3) only
    refines it. So their dip is an input to their FWI, not an output. Whether a
    single-well walkaway DAS-VSP can recover a dip AT ALL is unanswered in the
    literature and unanswerable on a 1-D model.
    Set a dip here, invert from a FLAT start, and measure what comes back.
    """
    z = np.arange(nz) * dz
    vp = np.empty(nz, dtype=np.float64)
    for i, zi in enumerate(z):
        if zi < z_air:
            vp[i] = V_AIR
        elif zi < Z_I_BOTTOM:
            f = (zi - z_air) / (Z_I_BOTTOM - z_air)
            vp[i] = V_I[0] + f * (V_I[1] - V_I[0])
        elif zi < Z_II_BOTTOM:
            f = (zi - Z_I_BOTTOM) / (Z_II_BOTTOM - Z_I_BOTTOM)
            vp[i] = V_II[0] + f * (V_II[1] - V_II[0])
        else:
            f = min((zi - Z_II_BOTTOM) / (V_III_BOTTOM_DEPTH - Z_II_BOTTOM), 1.0)
            vp[i] = V_III[0] + f * (V_III[1] - V_III[0])
    out = np.tile(vp[:, None], (1, nx))
    if dip_m_per_km:
        # dip_m_per_km is the CHANGE IN BOUNDARY DEPTH per km, so Park's sense
        # (basement rising to the right) is NEGATIVE. To place a boundary at
        # d0 + dip*x/1000 the profile must be sampled at z MINUS that amount --
        # the sign is easy to get backwards, and did on the first attempt: the
        # model came out slower to the right instead of faster.
        dxm = float(dx if dx is not None else dz)
        shift = -(np.arange(nx) * dxm / 1000.0) * float(dip_m_per_km)
        for j in range(nx):
            zs = z + shift[j]
            out[:, j] = np.interp(zs, z, vp)
            out[zs < z_air, j] = V_AIR          # keep the air slab intact
    return out


def gardner_rho(vp, rho_air=1.225, v_air_max=V_AIR + 1.0):
    """Gardner density 0.31*1000*vp^0.25, with air density WHERE vp is air.

    The air slab is identified by velocity (vp <= v_air_max), never by a fixed
    depth band: applying a low density under rock velocities destabilizes the
    FD scheme (dt/(rho*dx) blows up -> NaN wavefields, found in T7 bring-up).
    Models without an air layer (e.g. miniature tests) get pure Gardner.
    """
    vp = np.asarray(vp, dtype=np.float64)
    rho = 0.31 * 1000.0 * vp ** 0.25
    rho[vp <= v_air_max] = rho_air
    return rho


def make_acoustic_model(vp, vp_grad=False, dx=DX, dz=DZ, nabc=20,
                        free_surface=False, dtype=torch.float64,
                        device="cpu", vp_bound=None, rho=None):
    """AcousticModel wrapper with the project's fixed conventions.

    auto_update_rho=False ALWAYS: model.forward() would otherwise overwrite
    rho from vp through .data (no autograd path), silently desynchronizing
    AD and FD gradients (found in T4).

    rho: pass the SAME fixed density to the observed-data model and the
    inversion model. Defaulting rho to Gardner-of-this-vp is only safe when
    both models share the same vp; deriving obs-rho from vp_true but
    inversion-rho from vp_init puts a systematic amplitude error in the data
    that vp must absorb (the acoustic Marmousi demo's documented failure).
    """
    vp = np.asarray(vp, dtype=np.float64)
    nz, nx = vp.shape
    rho = gardner_rho(vp) if rho is None else np.asarray(rho, dtype=np.float64)
    return AcousticModel(0, 0, nx, nz, dx, dz, vp=vp, rho=rho,
                         vp_grad=vp_grad, rho_grad=False,
                         auto_update_rho=False, free_surface=free_surface,
                         abc_type="PML", nabc=nabc,
                         vp_bound=vp_bound, device=device, dtype=dtype)


def forge_fibers(nz, x_well_a=1000.0, x_well_b=1400.0, z_top=Z_AIR + 100.0,
                 n_channels=None, dz=DZ, synthetic=True, dx=None, gauge_l=None):
    """The two vertical FORGE fibers (wells 78A-32 and 78B-32).

    x positions default to placeholders; set the REAL well x-positions from
    the field survey at HPC stage. In synthetic mode channels sit exactly on
    grid nodes (inverse crime); n_channels defaults to filling the model down
    to ~85% of its depth.
    """
    if n_channels is None:
        n_channels = int(0.85 * nz - z_top / dz)
    # dx and the gauge length MUST track dz, or the fibres land in the wrong
    # place on any grid but the 5 m one this was written for. Two real faults,
    # both caught by the first 10 m run:
    #   * dx was hardcoded to the module DX (5 m), so on a 10 m grid every well
    #     x-position DOUBLED -- receiver index 367 on a 296-column model, which
    #     the elastic kernel reported as an out-of-bounds index.
    #   * the gauge length was hardcoded to 10 m = 2*5 m. E3 is EXACT only when
    #     the gauge endpoints sit on grid nodes, i.e. l = 2*dz; at dz = 10 m a
    #     10 m gauge puts them half a cell off and the operator stops being
    #     exact -- silently, since it still returns numbers.
    dx = float(DX if dx is None else dx)
    gauge_l = float(2.0 * dz if gauge_l is None else gauge_l)
    kwargs = dict(z_top=z_top, n_channels=n_channels, dch=DCH, l=gauge_l,
                  dx=dx, dz=dz, snap_to_nodes=not synthetic)
    return (FiberGeometry(x_well=x_well_a, **kwargs),
            FiberGeometry(x_well=x_well_b, **kwargs))


def vibroseis_line(nt, dt, f0, x_indices, z_index, amp0=1.0):
    """Surface vibroseis line: one Ricker source per x index.

    `z_index` may be a SCALAR (flat surface) or one depth PER SOURCE. Under
    topography the per-source form is required: with the measured FORGE ramp
    (162 m over 2960 m) a single median row buries 5 of 12 sources in the AIR
    layer, where they radiate at 340 m/s into nothing. The symptom is a NaN
    skip fraction, because the synthetic gather is empty.
    """
    src = Source(nt, dt, f0)
    wl = wavelet(nt, dt, f0, amp0=amp0)[1]
    zs = (np.full(len(x_indices), int(z_index)) if np.isscalar(z_index)
          else np.asarray(z_index, int))
    if len(zs) != len(x_indices):
        raise ValueError(f"z_index must be scalar or one per source, got "
                         f"{len(zs)} for {len(x_indices)} sources")
    for ix, iz in zip(x_indices, zs):
        src.add_source(int(ix), int(iz), wl)
    return src


def build_survey(source, geometry):
    """Survey whose receivers are the fiber's deduplicated gauge endpoints."""
    rcv = Receiver(source.nt, source.dt)
    rcv_z = np.array([kz for (kz, _kx) in geometry.rcv_pos])
    rcv_x = np.array([kx for (_kz, kx) in geometry.rcv_pos])
    rcv.add_receivers(rcv_x, rcv_z, "vz")   # add_receivers takes x FIRST
    return Survey(source, rcv)


def generate_observed(model, geometry, source, checkpoint_segments=10,
                      device="cpu", dtype=torch.float64):
    """Run the propagator on the TRUE model and record inverse-crime observed
    strain rate under the "strain_rate" key.

    Returns (obs_data, survey, layer): the SeismicData, the Survey it was
    recorded on (reuse it for the inversion propagator so the receiver layout
    is identical), and the DASObservationLayer (reuse as das_layer).
    """
    survey = build_survey(source, geometry)
    layer = DASObservationLayer(geometry, output="strain_rate")
    prop = AcousticPropagator(model, survey, device=device, dtype=dtype)
    with torch.no_grad():
        rec = prop.forward(checkpoint_segments=checkpoint_segments)
        obs_sr = layer(rec["u"], rec["w"])
    obs_data = SeismicData(survey)
    obs_data.record_data({"strain_rate": obs_sr})
    return obs_data, survey, layer


def marmousi2_proxy(in_dir, nx, nz, dx=DX, dz=DZ):
    """Marmousi2 vp resampled to our grid via ADFWI's own loader (downloads
    SEGY files into in_dir on first use - NOT exercised by local tests).

    Returns vp [nz, nx] float64; pair it with a synthetic vertical fiber via
    FiberGeometry(snap_to_nodes=False) for the Marmousi2-fiber campaign.
    """
    from ADFWI.utils.velocityDemo import (load_marmousi_model,
                                          resample_marmousi_model)
    marmousi = load_marmousi_model(in_dir)
    x = np.linspace(0, (nx - 1) * dx, nx)
    z = np.linspace(0, (nz - 1) * dz, nz)
    resampled = resample_marmousi_model(x, z, marmousi)
    return np.asarray(resampled["vp"], dtype=np.float64).T \
        if resampled["vp"].shape != (nz, nx) else \
        np.asarray(resampled["vp"], dtype=np.float64)
