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


def forge_proxy_vp(nz, nx, dz=DZ, z_air=Z_AIR):
    """Piecewise-linear FORGE proxy vp [nz, nx] (float64), zones per E11.

    The profile is 1-D in depth (laterally homogeneous) and broadcast over x.
    Zone III grades linearly from V_III[0] at Z_II_BOTTOM to V_III[1] at
    V_III_BOTTOM_DEPTH and stays at V_III[1] below that.
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
    return np.tile(vp[:, None], (1, nx))


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
                 n_channels=None, dz=DZ, synthetic=True):
    """The two vertical FORGE fibers (wells 78A-32 and 78B-32).

    x positions default to placeholders; set the REAL well x-positions from
    the field survey at HPC stage. In synthetic mode channels sit exactly on
    grid nodes (inverse crime); n_channels defaults to filling the model down
    to ~85% of its depth.
    """
    if n_channels is None:
        n_channels = int(0.85 * nz - z_top / dz)
    kwargs = dict(z_top=z_top, n_channels=n_channels, dch=DCH, l=GAUGE_L,
                  dx=DX, dz=dz, snap_to_nodes=not synthetic)
    return (FiberGeometry(x_well=x_well_a, **kwargs),
            FiberGeometry(x_well=x_well_b, **kwargs))


def vibroseis_line(nt, dt, f0, x_indices, z_index, amp0=1.0):
    """Surface vibroseis line: one Ricker source per x index at z_index
    (place z_index at/below the air-ground interface node z_air/dz)."""
    src = Source(nt, dt, f0)
    wl = wavelet(nt, dt, f0, amp0=amp0)[1]
    for ix in x_indices:
        src.add_source(int(ix), int(z_index), wl)
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
