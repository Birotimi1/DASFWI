"""Load the FORGE walkaway-VSP DAS strain-rate field data for E3 inversion.

The two monitoring wells 78A-32 (1010 channels) and 78B-32 (1206 channels)
are VERTICAL (confirmed from the SEG-Y: constant surface x,y down the trace
axis; TVD change per channel = 1.021 m = the 1.02 m channel spacing). The
field observable is strain rate natively (EBCDIC: "AMPLITUDE VALUES
PROPORTIONAL TO STRAINRATE; SC=X11600 TO NM/M/S"), so the existing vertical
``FiberGeometry`` + ``DASObservationLayer`` (E3 endpoint difference) is the
exact and complete operator - no strain->velocity conversion, no E5.

Acquisition: a "walkaway" VSP means the SOURCE walks away from the well
(318 surface shots per well, offsets 36-1547 m); the fiber is fixed. Each
``.sgy`` file is one shot recorded on all fiber channels.

WHAT THIS MODULE DOES
---------------------
1. Reads per-shot source (x,y,z) and the shared receiver (channel) geometry
   from the SEG-Y trace headers, applying the SEG-Y scalars.
2. Projects the 3-D UTM geometry onto a 2-D section for ADFWI's 2-D acoustic
   code. The section axis is the PRINCIPAL horizontal axis of the source
   cloud (PCA), passing through the wellhead; local x = signed distance
   along that axis from the wellhead, local z = depth below a surface datum.
   The out-of-plane component of the source spread is dropped - the standard,
   unavoidable approximation when inverting 3-D acquisition with a 2-D code;
   documented here so it is never silent.
3. Selects, for each inversion-grid depth node, the nearest physical fiber
   channel (the spec channel-subset rule, position error <= dch/2), giving an
   observed gather aligned 1:1 with the geometry's channels.
4. Resamples the observed traces from the field dt (1 ms) to the modelling dt
   when they differ (a 5 m grid needs dt <= ~0.4 ms for CFL stability at
   ~5900 m/s, so the 1 ms field data must be resampled for a 5 m run; a
   coarse smoke grid with dz >= ~12 m is stable at 1 ms and skips it).
5. Builds an ADFWI ``Survey`` (sources as grid indices with a PLACEHOLDER
   Ricker wavelet - the true field source wavelet is unknown and its
   estimation is a separate downstream step) plus the ``DASObservationLayer``,
   and returns everything needed to construct
   ``AcousticFWI(..., das_layer=layer, obs_key="strain_rate")``.

Nothing here reads a true model - field FWI has none; RMS-vs-truth metrics are
meaningless and the run scripts start from a 1-D gradient.
"""

import glob
import os
from pathlib import Path

import numpy as np
import torch

import segyio
from segyio import TraceField as TF

from ADFWI.survey import Source, Receiver, Survey, SeismicData
from ADFWI.utils import wavelet
from scipy import integrate, signal

from das.geometry import FiberGeometry
from das.das_layer import DASObservationLayer

# Default field-data location (LOCAL ONLY - never committed to git).
# Layouts supported, in order:
#   1. FORGE_DAS_DIR env var (always wins)
#   2. side-by-side:  <repo-parent>/DAS_VSP     (the HPC layout: data copied
#      next to DASFWI/, same convention as Data_downloads -- what
#      hpc/condor/fs_check.sub verifies)
#   3. grandparent:   <repo-grandparent>/DAS_VSP (the local dev layout:
#      repo = .../Codes/DASFWI, data = .../2022_Stimulation/DAS_VSP)
_REPO = Path(__file__).resolve().parents[1]


def _default_das_dir() -> Path:
    side_by_side = _REPO.parent / "DAS_VSP"
    if side_by_side.is_dir():
        return side_by_side
    return _REPO.parents[1] / "DAS_VSP"


DAS_VSP_DIR = Path(os.environ.get("FORGE_DAS_DIR", _default_das_dir()))

WELLS = ("78A-32", "78B-32")
FIELD_DT = 1e-3          # SEG-Y sample interval (s)
DCH = 1.02               # physical channel spacing (m)


# --------------------------------------------------------------------------- #
# SEG-Y geometry + data reading
# --------------------------------------------------------------------------- #
def _scalar(value, scal):
    """Apply a SEG-Y coordinate/elevation scalar (negative => divide)."""
    if scal < 0:
        return value / abs(scal)
    if scal > 0:
        return value * scal
    return value


def read_shot_geometry(well_dir, n_shots=None):
    """Read source (x,y,z) per shot and the shared receiver (x,y,z) column.

    Returns a dict with UTM/elevation arrays (metres, scalars applied):
        files      : list[str]           sorted shot file paths (len S)
        src_xyz    : [S, 3]              source X, Y, surface elevation
        rcv_xyz    : [C_phys, 3]         channel X, Y, group elevation
        nt, dt     : int, float          samples and sample interval (s)
    """
    files = sorted(glob.glob(os.path.join(str(well_dir), "*.sgy")))
    if not files:
        raise FileNotFoundError(f"no .sgy files under {well_dir}")
    if n_shots is not None:
        files = files[:n_shots]

    # receiver column (shared) + sampling, from the first shot
    with segyio.open(files[0], "r", ignore_geometry=True) as s:
        n_chan = s.tracecount
        nt = len(s.samples)
        dt = segyio.tools.dt(s) / 1e6                     # us -> s
        h0 = s.header[0]
        cscal, escal = h0[TF.SourceGroupScalar], h0[TF.ElevationScalar]
        gx = np.array([s.header[i][TF.GroupX] for i in range(n_chan)], float)
        gy = np.array([s.header[i][TF.GroupY] for i in range(n_chan)], float)
        gz = np.array([s.header[i][TF.ReceiverGroupElevation]
                       for i in range(n_chan)], float)
    rcv_xyz = np.stack([_scalar(gx, cscal), _scalar(gy, cscal),
                        _scalar(gz, escal)], axis=1)

    src = np.empty((len(files), 3), float)
    for j, f in enumerate(files):
        with segyio.open(f, "r", ignore_geometry=True) as s:
            h = s.header[0]
            src[j] = [_scalar(h[TF.SourceX], h[TF.SourceGroupScalar]),
                      _scalar(h[TF.SourceY], h[TF.SourceGroupScalar]),
                      _scalar(h[TF.SourceSurfaceElevation],
                              h[TF.ElevationScalar])]
    return dict(files=files, src_xyz=src, rcv_xyz=rcv_xyz, nt=nt, dt=dt)


def project_to_2d(src_xyz, rcv_xyz):
    """Project 3-D UTM geometry onto a 2-D section (see module docstring).

    Local frame: x-axis = principal horizontal axis of the source cloud
    (PCA), origin at the wellhead (the receiver column's surface x,y);
    z = depth below the highest source elevation (datum).

    Returns dict:
        src_x [S], src_z [S]   source local horizontal / depth (m)
        chan_z [C_phys]        channel depths (m, positive down)
        well_x (float)         well local horizontal position (m)
        axis [2], datum (float), out_of_plane [S] (dropped component, m)
    """
    well_xy = rcv_xyz[:, :2].mean(axis=0)                 # constant column
    sxy = src_xyz[:, :2]
    # principal horizontal axis of the source cloud
    d = sxy - sxy.mean(axis=0)
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    axis = vt[0]
    axis = axis / np.linalg.norm(axis)

    def along(p):
        return (p - well_xy) @ axis                       # signed, wellhead=0

    def perp(p):
        n = np.array([-axis[1], axis[0]])
        return (p - well_xy) @ n

    datum = src_xyz[:, 2].max()                           # highest source
    return dict(
        src_x=along(sxy), src_z=datum - src_xyz[:, 2],
        chan_z=datum - rcv_xyz[:, 2], well_x=0.0,
        axis=axis, datum=float(datum), out_of_plane=perp(sxy))


def load_strain_gathers(files, n_chan):
    """Load raw strain-rate gathers [S, nt_field, C_phys] (float32).

    Values are proportional to strain rate (nm/m/s up to the header scale
    factor); absolute calibration is unnecessary for normalized/correlation
    misfits and is left to a downstream step.
    """
    out = []
    for f in files:
        with segyio.open(f, "r", ignore_geometry=True) as s:
            g = segyio.tools.collect(s.trace[:]).astype(np.float32)  # [C, nt]
        out.append(g.T)                                   # -> [nt, C]
    return np.stack(out, axis=0)                           # [S, nt, C]


def _resample_time(gathers, dt_in, dt_out, nt_out):
    """Resample along time dt_in -> dt_out (polyphase) and crop/pad to nt_out."""
    if abs(dt_in - dt_out) < 1e-12:
        g = gathers
    else:
        # rational ratio dt_in/dt_out (e.g. 1.0/0.4 = 5/2)
        from fractions import Fraction
        r = Fraction(dt_in / dt_out).limit_denominator(1000)
        g = signal.resample_poly(gathers, r.numerator, r.denominator, axis=1)
    S, nt, C = g.shape
    if nt >= nt_out:
        return g[:, :nt_out, :]
    pad = np.zeros((S, nt_out - nt, C), g.dtype)
    return np.concatenate([g, pad], axis=1)


# --------------------------------------------------------------------------- #
# top-level loader
# --------------------------------------------------------------------------- #
def load_forge_field(well="78A-32", n_shots=20, dz=5.0, dx=5.0,
                     nt_model=2000, dt_model=4e-4, f0=15.0, nabc=30,
                     pad_nodes=15, device="cpu", das_dir=None):
    """Load a FORGE well into an inversion-ready bundle.

    Args:
        well: "78A-32" or "78B-32".
        n_shots: number of walkaway shots to load (subset for a first run).
        dz, dx: inversion grid spacing (m). 5 m is the FORGE spec; the gauge
            length is tied to the grid as l = 2*dz so gauge endpoints fall on
            nodes (10 m at dz=5 m, matching the real fiber). A coarse dz for
            fast smoke tests scales the gauge with it.
        nt_model, dt_model: modelling time samples and interval. dt_model must
            satisfy CFL for dz (<= ~0.4 ms at 5 m / 5900 m/s); field dt is
            1 ms, so observed data is resampled when dt_model != 1 ms.
        f0: PLACEHOLDER Ricker centre frequency (the true source is unknown).
        nabc, pad_nodes: absorbing border and extra grid padding.
        device: torch device for the layer.

    Returns dict:
        obs_data (SeismicData with key "strain_rate"), survey, das_layer,
        geometry, grid (nx, nz, dx, dz, nt, dt), well_x_index,
        channel_z_grid, projection info, and n_shots.
    """
    das_dir = Path(das_dir) if das_dir else DAS_VSP_DIR
    well_dir = das_dir / well
    gauge_l = 2.0 * dz

    geo = read_shot_geometry(well_dir, n_shots=n_shots)
    proj = project_to_2d(geo["src_xyz"], geo["rcv_xyz"])
    S = len(geo["files"])

    # ---- grid sizing from the projected geometry ----
    all_x = np.concatenate([proj["src_x"], [proj["well_x"]]])
    x_shift = pad_nodes * dx - all_x.min()
    src_x_grid = np.round((proj["src_x"] + x_shift) / dx).astype(int)
    well_x_grid = int(round((proj["well_x"] + x_shift) / dx))
    nx = int(np.ceil((all_x.max() + x_shift) / dx)) + pad_nodes

    chan_z = proj["chan_z"]                                # [C_phys], down
    z_bottom = chan_z.max()
    nz = int(np.ceil(z_bottom / dz)) + pad_nodes
    src_z_grid = np.clip(np.round(proj["src_z"] / dz).astype(int), 1, nz - 2)

    # ---- fiber geometry on the grid (field mode: nearest channel per node) ----
    z_top = float(chan_z.min())
    n_phys = len(chan_z)
    geometry = FiberGeometry(x_well=well_x_grid * dx, z_top=z_top,
                             n_channels=n_phys, dch=DCH, l=gauge_l,
                             dx=dx, dz=dz, snap_to_nodes=True)
    # map each geometry channel (a snapped node depth) to the nearest REAL
    # physical field channel, to pull its observed trace
    chan_idx = np.abs(chan_z[:, None]
                      - geometry.channel_z[None, :]).argmin(axis=0)

    # ---- observed data: load, channel-select, time-resample ----
    raw = load_strain_gathers(geo["files"], geo["rcv_xyz"].shape[0])  # [S,ntf,Cp]
    obs_sel = raw[:, :, chan_idx]                          # [S, ntf, C]
    obs = _resample_time(obs_sel, geo["dt"], dt_model, nt_model)  # [S,nt,C]
    obs = torch.from_numpy(np.ascontiguousarray(obs)).to(torch.float32)

    # ---- survey: real source grid positions + placeholder Ricker ----
    _, wav = wavelet(nt_model, dt_model, f0, amp0=1)
    wav = integrate.cumtrapz(wav, axis=-1, initial=0)     # Liu's mt convention
    source = Source(nt=nt_model, dt=dt_model, f0=f0)
    for j in range(S):
        source.add_source(src_x=int(src_x_grid[j]), src_z=int(src_z_grid[j]),
                          src_wavelet=wav, src_type="mt",
                          src_mt=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    rcv = Receiver(nt=nt_model, dt=dt_model)
    rcv_z = np.array([kz for (kz, _kx) in geometry.rcv_pos])
    rcv_x = np.array([kx for (_kz, kx) in geometry.rcv_pos])
    rcv.add_receivers(rcv_x, rcv_z, "vz")
    survey = Survey(source, rcv)

    layer = DASObservationLayer(geometry, output="strain_rate")
    layer = layer.to(torch.float32).to(device)

    obs_data = SeismicData(survey)
    obs_data.record_data({"strain_rate": obs})

    return dict(
        obs_data=obs_data, survey=survey, das_layer=layer, geometry=geometry,
        grid=dict(nx=nx, nz=nz, dx=dx, dz=dz, nt=nt_model, dt=dt_model,
                  nabc=nabc, gauge_l=gauge_l),
        well=well, n_shots=S, well_x_index=well_x_grid,
        channel_z_grid=geometry.channel_z, src_x_grid=src_x_grid,
        src_z_grid=src_z_grid, projection=proj, obs_shape=tuple(obs.shape))


def summarize(bundle):
    """One-line-per-item human summary of a loaded bundle."""
    g = bundle["grid"]
    p = bundle["projection"]
    lines = [
        f"well {bundle['well']}: {bundle['n_shots']} shots, "
        f"obs {bundle['obs_shape']} (strain rate)",
        f"grid nx={g['nx']} nz={g['nz']} dx={g['dx']} dz={g['dz']} "
        f"nt={g['nt']} dt={g['dt']*1e3:.2f} ms gauge_l={g['gauge_l']} m",
        f"well at x-index {bundle['well_x_index']} "
        f"(x={bundle['well_x_index']*g['dx']:.0f} m); "
        f"channels {bundle['channel_z_grid'].min():.0f}-"
        f"{bundle['channel_z_grid'].max():.0f} m depth ({bundle['geometry'].C})",
        f"source x-index {bundle['src_x_grid'].min()}-"
        f"{bundle['src_x_grid'].max()}, z-index "
        f"{bundle['src_z_grid'].min()}-{bundle['src_z_grid'].max()}",
        f"2-D projection axis {p['axis'].round(3).tolist()}, "
        f"out-of-plane source spread +-{np.abs(p['out_of_plane']).max():.0f} m "
        f"(dropped by the 2-D code)",
    ]
    return "\n".join(lines)
