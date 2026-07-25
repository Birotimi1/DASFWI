"""Frequency-adaptive objective: L2 at low frequency -> optimal transport high.

THE IDEA
--------
Cycle skipping bites when |dt| > T/2 = 1/(2 f_max), so the risk grows with
frequency. Where there is no skipping, plain L2 is the maximum-likelihood
estimator and gives the highest resolution (confirmed on this project: l2_adam
is the best acoustic combo, SSIM 0.868). Where skipping sets in, transport-type
misfits (Wasserstein-Sinkhorn) are convex in time shift and keep converging.
So we ramp continuously between them as the multiscale cascade climbs:

    E(lambda) = (1 - lambda) * Ehat_lo  +  lambda * Ehat_hi

with lambda = 0 at the lowest band and -> 1 as the band approaches the
skip-prone frequencies.

WHY THE TERMS MUST BE GRADIENT-NORMALISED (measured, not assumed)
-----------------------------------------------------------------
On real DAS strain-rate amplitudes (~2.3e-8) the two objectives are wildly
different in both value and gradient:

    misfit     value        |dE/dsyn|
    l2         5.64e-07     4.90e-01
    gc        -5.16e-02     2.55e+05      <-- NEGATIVE
    sinkhorn   1.31e-04     1.32e+03

The VALUE ratio (sinkhorn/l2) is 233 but the GRADIENT ratio is 2686 - a factor
11.5 apart - so normalising by the loss value does NOT equalise the influence on
the model update, and a sign-indefinite misfit (gc) breaks value-normalisation
outright. We therefore normalise each term by the norm of its ADJOINT SOURCE
dE/dsyn, taken as a DETACHED scalar. Dividing E by it makes each term contribute
a unit-norm adjoint source, so lambda controls the balance that actually reaches
the model.

This is also CHEAP: differentiating w.r.t. the synthetic data only walks the
small syn->E subgraph; the expensive propagator graph is never traversed. The
model gradient of the blend is then just the usual J^T applied to the blended
(already balanced) adjoint source.

Short-circuiting: sinkhorn costs ~20x L2 per call, so at lambda == 0 the high
term is never evaluated, and at lambda == 1 the low term is never evaluated.
"""
import math

import torch

from ADFWI.fwi.misfit import Misfit


# --------------------------------------------------------------------------- #
# the blended objective
# --------------------------------------------------------------------------- #
class BlendedMisfit(Misfit):
    """Convex blend of two misfits, balanced by adjoint-source norm.

    Args:
        loss_lo: the low-frequency / high-resolution misfit (e.g. L2).
        loss_hi: the robust / cycle-skip-tolerant misfit (e.g. sinkhorn).
        lam: initial blend weight in [0, 1] (0 = pure lo, 1 = pure hi).
        beta: EMA factor for the normalising scales (0 disables smoothing).
        normalize: if False, blend the RAW losses (diagnostic only - lambda then
            does not mean what you think; see the module docstring).

    Call convention: `forward(a, b)` passes both arguments through POSITIONALLY
    to the sub-misfits, so whatever convention the caller uses is preserved.
    (ADFWI's AcousticFWI calls `loss_fn.forward(synthetic, observed)` even though
    most upstream misfits *name* the first parameter `obs`.) The normalising
    gradient is taken w.r.t. whichever argument carries the autograd graph.
    """

    def __init__(self, loss_lo, loss_hi, lam=0.0, beta=0.9, normalize=True,
                 eps=1e-30):
        super().__init__()
        self.loss_lo, self.loss_hi = loss_lo, loss_hi
        self.beta, self.normalize, self.eps = beta, normalize, eps
        self._ema = {"lo": None, "hi": None}
        self.set_lambda(lam)

    # -- schedule control ---------------------------------------------------
    def set_lambda(self, lam):
        if not (0.0 <= float(lam) <= 1.0):
            raise ValueError(f"lambda must be in [0,1], got {lam}")
        self.lam = float(lam)
        return self.lam

    @property
    def scales(self):
        """Current EMA normalising scales (diagnostics/logging)."""
        return dict(self._ema)

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _graph_arg(a, b):
        """Whichever input carries the autograd graph (the synthetic)."""
        if torch.is_tensor(a) and a.requires_grad:
            return a
        if torch.is_tensor(b) and b.requires_grad:
            return b
        return None

    def _scale(self, key, value, wrt):
        """Detached ||dE/dsyn||, EMA-smoothed. Falls back to |E| when there is
        no graph (e.g. unit tests on plain tensors)."""
        if wrt is None:
            s = float(value.detach().abs()) if torch.is_tensor(value) else abs(float(value))
        else:
            g, = torch.autograd.grad(value, wrt, retain_graph=True,
                                     create_graph=False, allow_unused=True)
            s = float(g.detach().norm()) if g is not None else 1.0
        s = max(s, self.eps)
        prev = self._ema[key]
        s = s if (prev is None or self.beta <= 0) else self.beta * prev + (1 - self.beta) * s
        self._ema[key] = s
        return s

    # -- objective ----------------------------------------------------------
    @staticmethod
    def _eval(fn, a, b):
        """Evaluate a sub-misfit. Uses apply_misfit so that NIM (an
        autograd.Function invoked via .apply) works as a term too."""
        from inversion.safe_misfits import apply_misfit
        return apply_misfit(fn, a, b)

    def forward(self, a, b):
        wrt = self._graph_arg(a, b)
        lam = self.lam

        if lam <= 0.0:                                   # short-circuit: lo only
            e = self._eval(self.loss_lo, a, b)
            return e / self._scale("lo", e, wrt) if self.normalize else e
        if lam >= 1.0:                                   # short-circuit: hi only
            e = self._eval(self.loss_hi, a, b)
            return e / self._scale("hi", e, wrt) if self.normalize else e

        e_lo = self._eval(self.loss_lo, a, b)
        e_hi = self._eval(self.loss_hi, a, b)
        if not self.normalize:
            return (1.0 - lam) * e_lo + lam * e_hi
        return ((1.0 - lam) * e_lo / self._scale("lo", e_lo, wrt)
                + lam * e_hi / self._scale("hi", e_hi, wrt))


# --------------------------------------------------------------------------- #
# lambda schedules
# --------------------------------------------------------------------------- #
class LambdaSchedule:
    """lambda(frequency, stage) for the multiscale cascade.

    Frequency mode (default): lambda ramps log-linearly from 0 at `f_lo` to 1 at
    `f_hi`. Set (f_lo, f_hi) from the PHASE 1 flip curve - f_lo is the highest
    band where L2 still wins, f_hi the band where the robust misfit is clearly
    ahead. Until Phase 1 has run these are placeholders, NOT physics.

    Per-stage overrides exist because frequency is not the only skip risk. When
    Vs is first released its starting model comes from a Vp/sqrt(3) guess, which
    in a sedimentary cover (true Vp/Vs ~ 2.2) can be ~200 ms off - beyond T/2 at
    3 Hz. So the Vs-release stage starts robust (lambda = 1) regardless of band
    and anneals down over `stage_anneal` bands.
    """

    def __init__(self, f_lo, f_hi, stage_overrides=None, stage_anneal=1):
        if not (f_hi > f_lo > 0):
            raise ValueError(f"need f_hi > f_lo > 0, got {f_lo}, {f_hi}")
        self.f_lo, self.f_hi = float(f_lo), float(f_hi)
        self.stage_overrides = dict(stage_overrides or {})
        self.stage_anneal = max(1, int(stage_anneal))

    def lam(self, f_band, stage=None, bands_since_stage_start=0):
        """Blend weight for this band (and optionally this parameter stage)."""
        t = ((math.log(max(f_band, 1e-9)) - math.log(self.f_lo))
             / (math.log(self.f_hi) - math.log(self.f_lo)))
        lam = min(1.0, max(0.0, t))
        if stage in self.stage_overrides:
            start = float(self.stage_overrides[stage])
            # anneal from the override toward the frequency schedule
            w = min(1.0, bands_since_stage_start / self.stage_anneal)
            lam = (1.0 - w) * start + w * lam
        return lam

    def __repr__(self):
        return (f"LambdaSchedule(f_lo={self.f_lo}, f_hi={self.f_hi}, "
                f"stage_overrides={self.stage_overrides})")


class DiagnosticLambda(LambdaSchedule):
    """PHASE 5: lambda driven by the MEASURED cycle-skip fraction, with the
    frequency schedule as a prior and hysteresis so it cannot oscillate.

    Rationale: frequency is only a proxy for skip risk. A good starting model can
    carry L2 safely into high frequency, and a purely frequency-scheduled lambda
    would needlessly sacrifice L2's resolution there. This variant raises lambda
    only when skipping is actually observed.

    `on_above` / `off_below` are skip fractions; the gap between them is the
    hysteresis band (an observation inside it leaves lambda unchanged).
    """

    def __init__(self, f_lo, f_hi, on_above=0.30, off_below=0.15,
                 stage_overrides=None, stage_anneal=1, blend=0.5):
        super().__init__(f_lo, f_hi, stage_overrides, stage_anneal)
        if not (0.0 <= off_below < on_above <= 1.0):
            raise ValueError("need 0 <= off_below < on_above <= 1")
        self.on_above, self.off_below, self.blend = on_above, off_below, float(blend)
        self._state = 0.0            # last committed diagnostic-driven weight

    def lam(self, f_band, stage=None, bands_since_stage_start=0,
            skip_fraction=None):
        prior = super().lam(f_band, stage, bands_since_stage_start)
        if skip_fraction is None or not math.isfinite(float(skip_fraction)):
            return prior
        s = float(skip_fraction)
        if s >= self.on_above:                     # skipping observed -> robust
            self._state = 1.0
        elif s <= self.off_below:                  # well aligned -> resolution
            self._state = 0.0
        # inside the hysteresis band: keep the previous state
        return (1.0 - self.blend) * prior + self.blend * self._state


def stage_plan(bands, schedule, f_max_source, vs_release_band=None):
    """Resolve the 2-D (band x parameter-stage) schedule into a plan.

    Args:
        bands: list of low-pass cut-offs; None entries mean "full band".
        schedule: a LambdaSchedule (or None for a fixed-misfit control arm).
        f_max_source: the source's own f_max (f90), used for the 'full' band and
            as a ceiling - filtering cannot raise the frequency content.
        vs_release_band: 1-based band where Vs joins the inversion (None = Vp
            only throughout, e.g. the acoustic driver).

    Returns a list of dicts: band (1-based), cutoff, f_eff, stage, vs_live, lam.
    """
    plan = []
    for bi, cut in enumerate(bands, start=1):
        f_eff = f_max_source if cut is None else min(float(cut), f_max_source)
        vs_live = (vs_release_band is not None and bi >= vs_release_band)
        stage = "vs" if vs_live else "vp"
        lam = None
        if schedule is not None:
            since = 0 if vs_release_band is None else max(0, bi - vs_release_band)
            lam = schedule.lam(f_eff, stage=stage if vs_live else None,
                               bands_since_stage_start=since)
        plan.append(dict(band=bi, cutoff=cut, f_eff=f_eff, stage=stage,
                         vs_live=vs_live, lam=lam))
    return plan


def build_adaptive_misfit(name_lo, name_hi, dt, iterations, lam=0.0, **kw):
    """Convenience: build a BlendedMisfit from two registry names."""
    from inversion import config
    return BlendedMisfit(config.build_misfit(name_lo, dt=dt, iterations=iterations),
                         config.build_misfit(name_hi, dt=dt, iterations=iterations),
                         lam=lam, **kw)
