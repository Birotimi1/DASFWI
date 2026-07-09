"""T4 correctness gates E6-E9 (build spec section 2).

E6  identity (layer alone): plane-wave analytic identity, <= 1e-12 relative.
E7  dot-product/adjoint (layer alone): <R(v), y> == <v, R^T y> to 1e-12.
E8  AD-vs-FD gradient through the FULL chain (propagator -> DAS layer -> GC
    misfit): per-cell relative error <= 1e-3. This extends ADFWI's published
    AD-vs-FD validation THROUGH our observation operator - the project's key
    correctness gate.
E9  arrival sanity: gauge-channel first break lies between its endpoint
    receivers' first breaks; single-cell center estimate agrees within
    (l/2)/v_local + 2 samples.

All float64, all CPU. The tiny setup: homogeneous vp = 2000 m/s, nz = nx = 101,
dx = dz = 5 m, one shot (Ricker f0 = 10 Hz) near the surface, nt = 1500,
dt = 0.4 ms, vertical fiber at an offset x-node with channels on grid nodes
(synthetic geometry mode).

ADFWI APIs confirmed by source read (spec rule 2):
- AcousticModel(ox, oz, nx, nz, dx, dz, vp=, rho=, vp_grad=, rho_grad=,
  auto_update_rho=, free_surface=, abc_type=, nabc=, device=, dtype=)
  [model/acoustic_model.py L21]. auto_update_rho=False here: model.forward()
  would otherwise overwrite rho from vp via .data (no autograd path), which
  would make FD (sees the rho change) disagree with AD (does not).
- Source(nt, dt, f0).add_source(src_x, src_z, src_wavelet) [survey/source.py];
  utils.wavelets.wavelet(nt, dt, f0) returns (t, wavelet) - take [1].
- Receiver(nt, dt).add_receivers(rcv_x, rcv_z, rcv_type) - x FIRST
  [survey/receiver.py L70]; our FiberGeometry.rcv_pos is (z_index, x_index).
- Survey(source, receiver) [survey/survey.py].
- AcousticPropagator(model, survey, device=, dtype=) [acoustic_propagator.py
  L21]; forward(shot_index=None, checkpoint_segments=) returns dict with
  "p"/"u"/"w" records [n_shot, nt, n_rcv]; kernel offsets src/rcv indices by
  nabc internally [acoustic_kernels.py L248-252]; the update equations show
  u is driven by dp/dx (horizontal) and w by dp/dz (vertical) - the kernel
  DOCSTRING labels them backwards; trust the equations.
- Misfit_global_correlation(dt=1).forward(obs, syn) -> scalar loss
  [fwi/misfit/GlobalCorrelation.py L13]. UPSTREAM BUG (documented, file is
  read-only per spec rule 3): its forward() allocates the residual accumulator
  as `torch.zeros(..., device=...)` WITHOUT dtype, i.e. float32, so float64
  inputs crash with an index_put dtype mismatch. Stock ADFWI never sees this
  (it runs float32). The float64 gates below use GCMisfit64, a subclass whose
  forward is the same math with the accumulator dtype following the inputs.
"""

import numpy as np
import pytest
import torch

from ADFWI.model import AcousticModel
from ADFWI.survey import Source, Receiver, Survey
from ADFWI.propagator import AcousticPropagator
from ADFWI.utils.wavelets import wavelet
from ADFWI.fwi.misfit import Misfit_global_correlation

from das.geometry import FiberGeometry
from das.das_layer import DASObservationLayer


class GCMisfit64(Misfit_global_correlation):
    """Misfit_global_correlation with the accumulator dtype following the
    inputs (upstream hard-codes float32; see module docstring). Math identical.
    """

    def forward(self, obs, syn):
        mask1 = torch.sum(torch.abs(obs), axis=1) == 0
        mask2 = torch.sum(torch.abs(syn), axis=1) == 0
        mask = ~(mask1 * mask2)

        rsd = torch.zeros((obs.shape[0], obs.shape[2]),
                          device=obs.device, dtype=obs.dtype)
        for itrace in range(obs.shape[2]):
            shot_idx = torch.argwhere(mask[:, itrace])
            obs_trace = obs[shot_idx, :, itrace].squeeze(axis=1)
            syn_trace = syn[shot_idx, :, itrace].squeeze(axis=1)

            obs_trace = obs_trace / obs_trace.norm(dim=1, keepdim=True)
            syn_trace = syn_trace / syn_trace.norm(dim=1, keepdim=True)

            cov = torch.mean(obs_trace * syn_trace, dim=1)
            var_obs = torch.var(obs_trace, dim=1)
            var_syn = torch.var(syn_trace, dim=1)

            corr = cov / (torch.sqrt(var_obs * var_syn) + 1e-8)
            corr[torch.isnan(corr)] = 0
            rsd[shot_idx, itrace] = -corr.reshape(-1, 1)

        return torch.sum(rsd * self.dt)

# ----------------------------------------------------------------------------
# tiny setup constants (spec T4/E8)
# ----------------------------------------------------------------------------
NX = NZ = 101
DX = DZ = 5.0
NT = 1500
DT = 4e-4
F0 = 10.0
NABC = 20
VP0 = 2000.0
SRC_X, SRC_Z = 20, 2          # grid indices (unpadded); kernel adds nabc
X_WELL = 300.0                # x-node 60, well offset from the source
Z_TOP = 100.0                 # first channel depth (node 20)
N_CH = 20                     # channels on nodes 20..39 (synthetic mode)
FD_DELTA = 1.0                # m/s, central differences
CHECKPOINT_SEGMENTS = 10


def make_geometry():
    return FiberGeometry(x_well=X_WELL, z_top=Z_TOP, n_channels=N_CH,
                         dch=1.02, l=10.0, dx=DX, dz=DZ, snap_to_nodes=False)


def make_model(vp, vp_grad=False):
    rho = 0.31 * 1000.0 * np.asarray(vp) ** 0.25  # Gardner, FIXED (see header)
    return AcousticModel(0, 0, NX, NZ, DX, DZ,
                         vp=np.asarray(vp, dtype=np.float64), rho=rho,
                         vp_grad=vp_grad, rho_grad=False,
                         auto_update_rho=False, free_surface=False,
                         abc_type="PML", nabc=NABC,
                         device="cpu", dtype=torch.float64)


def make_survey(geometry):
    src = Source(NT, DT, F0)
    src.add_source(SRC_X, SRC_Z, wavelet(NT, DT, F0)[1])
    rcv = Receiver(NT, DT)
    rcv_z = np.array([kz for (kz, _kx) in geometry.rcv_pos])
    rcv_x = np.array([kx for (_kz, kx) in geometry.rcv_pos])
    rcv.add_receivers(rcv_x, rcv_z, "vz")   # add_receivers takes x FIRST
    return Survey(src, rcv)


def das_forward(prop, model, layer):
    """Full chain: propagator -> DAS layer -> strain-rate gather [S, T, C].

    The propagator is built ONCE and reused: its constructor derives the PML
    damping profile from vp.max(), so rebuilding it per perturbed model would
    make the absorbing boundary itself depend on the perturbation - a spurious
    J-dependence that autograd (fixed damp) correctly does not see. Passing
    model= to forward() keeps damp frozen, matching how AcousticFWI iterates.
    """
    rec = prop.forward(model=model, checkpoint_segments=CHECKPOINT_SEGMENTS)
    return layer(rec["u"], rec["w"]), rec


def first_break(trace, frac=0.05):
    """First sample index where |amp| exceeds frac * max|trace|."""
    a = trace.abs()
    return int((a > frac * a.max()).to(torch.int64).argmax())


# ----------------------------------------------------------------------------
# shared expensive runs (module scope)
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def geometry():
    return make_geometry()


@pytest.fixture(scope="module")
def layer(geometry):
    return DASObservationLayer(geometry, output="strain_rate")


@pytest.fixture(scope="module")
def propagator(geometry):
    """Single shared propagator (frozen PML damp; see das_forward docstring)."""
    return AcousticPropagator(make_model(np.full((NZ, NX), VP0)),
                              make_survey(geometry),
                              device="cpu", dtype=torch.float64)


@pytest.fixture(scope="module")
def obs_strain_rate(geometry, layer, propagator):
    """Inverse-crime observed data from a perturbed (bumped) true model."""
    zz, xx = np.meshgrid(np.arange(NZ), np.arange(NX), indexing="ij")
    bump = 100.0 * np.exp(-(((zz - 30.0) ** 2 + (xx - 40.0) ** 2) / (2 * 8.0 ** 2)))
    vp_true = np.full((NZ, NX), VP0) + bump
    with torch.no_grad():
        obs, _ = das_forward(propagator, make_model(vp_true), layer)
    return obs


@pytest.fixture(scope="module")
def homogeneous_run(geometry, layer, propagator):
    """No-grad run on the homogeneous model (E9 uses its w records)."""
    with torch.no_grad():
        syn, rec = das_forward(propagator,
                               make_model(np.full((NZ, NX), VP0)), layer)
    return syn, rec


# ----------------------------------------------------------------------------
# E6 - analytic identity, layer alone
# ----------------------------------------------------------------------------

def test_e6_identity(geometry, layer):
    c0 = 2000.0
    t = np.arange(600) * 5e-4

    def f(x):
        return np.exp(-((x - 0.35) / 0.05) ** 2)

    z_r = np.array([kz * DZ for (kz, _kx) in geometry.rcv_pos])
    v_e = f(t[:, None] - z_r[None, :] / c0)
    rcv_w = torch.from_numpy(v_e[None, ...].copy())
    rcv_u = torch.zeros_like(rcv_w)

    out = layer(rcv_u, rcv_w).numpy()[0]

    z_c = geometry.channel_z
    expected = (f(t[:, None] - (z_c[None, :] + geometry.l / 2) / c0)
                - f(t[:, None] - (z_c[None, :] - geometry.l / 2) / c0)) / geometry.l

    rel = np.abs(out - expected).max() / np.abs(expected).max()
    assert rel <= 1e-12, f"E6 identity relative error {rel:.3e} > 1e-12"


# ----------------------------------------------------------------------------
# E7 - dot-product/adjoint test, layer alone
# ----------------------------------------------------------------------------

def test_e7_dot_product_adjoint(geometry, layer):
    g = torch.Generator().manual_seed(42)
    S, T, R, C = 2, 64, geometry.n_rcv, geometry.C
    rcv_u = torch.randn(S, T, R, dtype=torch.float64, generator=g,
                        requires_grad=True)
    rcv_w = torch.randn(S, T, R, dtype=torch.float64, generator=g,
                        requires_grad=True)
    y = torch.randn(S, T, C, dtype=torch.float64, generator=g)

    a = (layer(rcv_u, rcv_w) * y).sum()
    g_u, g_w = torch.autograd.grad(a, (rcv_u, rcv_w))
    b = (rcv_u * g_u).sum() + (rcv_w * g_w).sum()

    rel = abs(float(a - b)) / (abs(float(a)) + abs(float(b)))
    assert rel <= 1e-12, f"E7 adjoint relative error {rel:.3e} > 1e-12"


# ----------------------------------------------------------------------------
# E8 - AD vs FD gradient through the full chain (THE key gate)
# ----------------------------------------------------------------------------

def test_e8_ad_vs_fd_gradient(geometry, layer, propagator, obs_strain_rate):
    misfit = GCMisfit64(dt=1)

    # AD gradient on the homogeneous starting model.
    model = make_model(np.full((NZ, NX), VP0), vp_grad=True)
    syn, _ = das_forward(propagator, model, layer)
    loss = misfit.forward(obs_strain_rate, syn)
    loss.backward()
    grad = model.vp.grad.detach().clone()
    assert torch.isfinite(grad).all(), "E8: AD gradient not finite"
    assert grad.abs().max() > 0, "E8: AD gradient identically zero"

    # Sample ~10 cells: 8 spread over the largest-|grad| region (away from the
    # absorbing boundary margin) + 2 fixed pseudo-random interior cells.
    margin = 10
    interior = grad[margin:-margin, margin:-margin].abs()
    flat_order = torch.argsort(interior.reshape(-1), descending=True)
    n_int_x = NX - 2 * margin
    cells = []
    for r in flat_order[:: max(1, len(flat_order) // 5000)]:
        iz = int(r) // n_int_x + margin
        ix = int(r) % n_int_x + margin
        if all(abs(iz - z0) + abs(ix - x0) > 8 for (z0, x0) in cells):
            cells.append((iz, ix))
        if len(cells) == 8:
            break
    rng = np.random.default_rng(7)
    while len(cells) < 10:
        iz = int(rng.integers(margin, NZ - margin))
        ix = int(rng.integers(margin, NX - margin))
        if all((iz, ix) != c for c in cells):
            cells.append((iz, ix))

    # Central finite differences (two no-grad forwards per cell).
    fd_model = make_model(np.full((NZ, NX), VP0))
    errors = {}
    for (iz, ix) in cells:
        j_pm = []
        for s in (+1.0, -1.0):
            with torch.no_grad():
                fd_model.vp.data[iz, ix] += s * FD_DELTA
                syn_fd, _ = das_forward(propagator, fd_model, layer)
                j_pm.append(float(misfit.forward(obs_strain_rate, syn_fd)))
                fd_model.vp.data[iz, ix] -= s * FD_DELTA
        fd = (j_pm[0] - j_pm[1]) / (2 * FD_DELTA)
        ad = float(grad[iz, ix])
        rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-300)
        errors[(iz, ix)] = (ad, fd, rel)

    report = "\n".join(
        f"  cell (z={iz:3d}, x={ix:3d}): AD={ad:+.6e}  FD={fd:+.6e}  rel={rel:.3e}"
        for (iz, ix), (ad, fd, rel) in errors.items())
    worst = max(rel for (_, _, rel) in errors.values())
    assert worst <= 1e-3, (
        f"E8: worst per-cell AD-vs-FD relative error {worst:.3e} > 1e-3\n{report}")


# ----------------------------------------------------------------------------
# E9 - arrival sanity
# ----------------------------------------------------------------------------

def test_e9_arrival_sanity(geometry, layer, homogeneous_run):
    syn, rec = homogeneous_run
    w = rec["w"][0]                     # [T, R]
    c = N_CH // 2                       # a mid-fiber channel

    fb_gauge = first_break(syn[0, :, c])
    fb_minus = first_break(w[:, int(geometry.idx_minus[c])])
    fb_plus = first_break(w[:, int(geometry.idx_plus[c])])

    lo, hi = min(fb_minus, fb_plus), max(fb_minus, fb_plus)
    assert lo <= fb_gauge <= hi, (
        f"E9: gauge first break {fb_gauge} not within endpoint first breaks "
        f"[{lo}, {hi}]")

    # N_g=1-style center estimate: single-cell difference of w over one dz at
    # the channel center (center node is receiver index idx_minus[c] + 1 in the
    # aligned case; the +1 node is idx_plus[c]).
    center = int(geometry.idx_minus[c]) + 1
    single_cell = (w[:, center + 1] - w[:, center]) / DZ
    fb_single = first_break(single_cell)

    tol_samples = (geometry.l / 2.0) / VP0 / DT + 2
    assert abs(fb_single - fb_gauge) <= tol_samples, (
        f"E9: single-cell first break {fb_single} vs gauge {fb_gauge} differs "
        f"by more than {tol_samples:.2f} samples")
