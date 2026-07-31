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


def _reject_stateful(side, fn):
    """Refuse a misfit that carries its OWN iteration schedule as a blend term.

    `weci` (ADFWI Misfit_weighted_ECI) is the case that matters: it is itself a
    hardcoded staged switch --
        w_i  = 1/(1+exp(-(self.iter - max_iter/2)));  loss = w_i*GC + (1-w_i)*ECI
        self.iter += 1     # once per forward CALL
    i.e. envelope(ECI) for the first half of max_iter, then a sigmoid hand-over
    to GLOBAL CORRELATION, advanced by call count. Inside a switching/blending
    controller that is silently wrong in both directions: BlendedMisfit's
    short-circuit skips the unused term, so its counter freezes and it never
    leaves the envelope end; and if it IS called past max_iter/2 the supposedly
    phase-INSENSITIVE robust term has quietly become phase-SENSITIVE GC. Its
    behaviour then depends on call history, which the controller cannot see.

    Use the stateless `envelope` (Misfit_envelope) as the robust term; the
    refinement half of weci's internal schedule is exactly what the L2 leg of
    the switch supplies, with skip-driven rather than hardcoded timing.
    """
    if hasattr(fn, "iter") and hasattr(fn, "max_iter"):
        raise ValueError(
            f"BlendedMisfit: {type(fn).__name__} (the '{side}' term) carries its "
            "own iteration schedule (self.iter/max_iter) and cannot be blended "
            "or switched -- see _reject_stateful. Use the stateless 'envelope' "
            "misfit as the robust term, or pass allow_stateful=True if you have "
            "externally pinned its weight.")


def _grad_scale(value, wrt, ema, key, beta, eps):
    """Detached ||dE/dsyn||, EMA-smoothed into ema[key]. Falls back to |E| when
    there is no graph (e.g. unit tests on plain tensors). Shared by
    BlendedMisfit and StagedMisfit so both normalise identically."""
    if wrt is None:
        s = float(value.detach().abs()) if torch.is_tensor(value) else abs(float(value))
    else:
        g, = torch.autograd.grad(value, wrt, retain_graph=True,
                                 create_graph=False, allow_unused=True)
        s = float(g.detach().norm()) if g is not None else 1.0
    s = max(s, eps)
    prev = ema.get(key)
    s = s if (prev is None or beta <= 0) else beta * prev + (1 - beta) * s
    ema[key] = s
    return s


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
                 eps=1e-30, allow_stateful=False):
        super().__init__()
        if not allow_stateful:
            for side, fn in (("lo", loss_lo), ("hi", loss_hi)):
                _reject_stateful(side, fn)
        self.loss_lo, self.loss_hi = loss_lo, loss_hi
        self.beta, self.normalize, self.eps = beta, normalize, eps
        self._ema = {"lo": None, "hi": None}
        self.set_lambda(lam)

    # -- schedule control ---------------------------------------------------
    def set_lambda(self, lam):
        if lam is None:
            # stage_plan returns lam=None when schedule=None (the skip-driven
            # path, where a controller sets lambda per chunk). Reaching here
            # means a band-level setter ran on that path -- guard the call site.
            raise ValueError(
                "set_lambda(None): no lambda to apply. Under skip-driven timing "
                "the controller sets lambda per chunk; the band-level call must "
                "be guarded with `if lam is not None`.")
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
        return _grad_scale(value, wrt, self._ema, key, self.beta, self.eps)

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


#: Phase-1 gate calibration (acoustic Marmousi, full band f90=6.25 Hz, 2026-07):
#: L2 wins at measured skip 0.51 (s6) and has collapsed by 0.64 (s16), where the
#: robust envelope family wins. The flip point is only constrained to the open
#: interval (0.51, 0.64), so: switch TO robust safely below 0.64, and hand back
#: to L2 only safely below 0.51 -- switching to L2 too early is the dangerous
#: direction (a robust term lingering at skip ~0.5 cost only ~0.05 SSIM in the
#: gate). off_below should be refined empirically by the hand-over sweep.
SKIP_ON_ABOVE = 0.58
SKIP_OFF_BELOW = 0.45


class SkipSwitch:
    """The verified STAGED misfit switch: a binary lambda in {0,1} driven by the
    MEASURED cycle-skip fraction. lambda=1 -> robust term (envelope), lambda=0
    -> resolution term (L2). Sits on top of BlendedMisfit, whose short-circuit
    never evaluates the unused term and whose detached grad-norm normalization
    keeps the adjoint-source norm ~unit across the hand-over (so Adam/adagrad
    moment states see no orders-of-magnitude rescale at the switch).

    Controller hygiene (each line fixes a failure mode found in verification):
      * state INITIALIZES from the FIRST measurement (a hardcoded lambda=0 start
        would run pure L2 whenever the start sits inside the hysteresis band --
        reproducing the exact failure the switch exists to prevent);
      * EMA smoothing of the skip series (one noisy measurement must not flip
        the mode);
      * a minimum DWELL (in updates) per mode -- no chatter at a threshold;
      * a hand-back RATCHET: at most `max_handbacks` robust->L2 transitions,
        and every attempted re-entry to robust afterwards is counted in
        `reentries` (>0 means the hand-over criterion is suspect -- abort/inspect).

    Call `update(skip)` once per measurement (e.g. per 25-iteration chunk); it
    returns the committed lambda. `history` records (n, skip_raw, lam).
    """

    def __init__(self, on_above=SKIP_ON_ABOVE, off_below=SKIP_OFF_BELOW,
                 ema=0.5, dwell=1, max_handbacks=1,
                 patience=2, min_progress=0.02, max_robust=None):
        if not (0.0 <= off_below < on_above <= 1.0):
            raise ValueError("need 0 <= off_below < on_above <= 1")
        self.on_above, self.off_below = float(on_above), float(off_below)
        self.ema, self.dwell = float(ema), int(dwell)
        self.max_handbacks = int(max_handbacks)
        # STALL GUARD. The robust stage earns its iterations by REDUCING skip; if
        # it does not, waiting longer cannot help and may actively harm. Measured
        # on elastic Marmousi from a 1-D linear start: skip sat at 0.63/0.55 --
        # never reaching off_below -- so the controller held envelope for all 200
        # iterations and the model DIVERGED (SSIM 0.361 -> 0.149). Skip alone is
        # not a sufficient signal: a robust stage that is degrading the model
        # keeps skip high, which keeps it selected. So force the hand-over after
        # `patience` consecutive robust updates that fail to improve the smoothed
        # skip by `min_progress`.
        self.patience, self.min_progress = int(patience), float(min_progress)
        self.max_robust = None if max_robust is None else int(max_robust)
        self._robust_start = None    # smoothed skip when robust mode began
        self._n_robust = 0           # updates spent in the current robust stage
        self.forced_handover = False # set if the stall guard fired
        self._smooth = None          # EMA of the skip series
        self._state = None           # committed lambda; None until first update
        self._age = 0                # updates since the last mode change
        self.handbacks = 0           # robust -> L2 transitions committed
        self.reentries = 0           # attempted L2 -> robust AFTER a handback
        self.history = []

    @property
    def lam(self):
        return 0.0 if self._state is None else self._state

    def update(self, skip):
        s = float(skip)
        if not math.isfinite(s):                   # broken measurement: hold
            self.history.append((len(self.history), s, self.lam))
            return self.lam
        self._smooth = s if self._smooth is None else \
            self.ema * s + (1.0 - self.ema) * self._smooth
        sm = self._smooth
        if self._state is None:
            # first measurement decides the starting mode; inside the band,
            # start ROBUST (late hand-over is cheap, early hand-over is not)
            self._state = 0.0 if sm <= self.off_below else 1.0
        else:
            # a robust-worthy signal AFTER a hand-back is the abort signal --
            # log the attempt unconditionally (dwell/ratchet may still block it)
            if self._state == 0.0 and sm >= self.on_above and self.handbacks > 0:
                self.reentries += 1
            if self._age >= self.dwell:
                if self._state == 0.0 and sm >= self.on_above \
                        and self.handbacks < self.max_handbacks:
                    self._state, self._age = 1.0, -1
                elif self._state == 1.0 and sm <= self.off_below:
                    self._state, self._age = 0.0, -1
                    self.handbacks += 1
        # STALL GUARD: at the observed rate, will robust reach off_below inside
        # the remaining budget? Per-step progress is the wrong test -- a slow
        # monotone drift (0.63 -> 0.60 over 3 updates) always "improves" yet
        # needs 15 more updates when the band only has 4.
        if self._state == 1.0:
            if self._robust_start is None:
                self._robust_start, self._n_robust = sm, 0
            self._n_robust += 1
            rate = (self._robust_start - sm) / max(self._n_robust, 1)
            left = (self.max_robust - self._n_robust) if self.max_robust else None
            # where will skip be if the remaining robust budget keeps this rate?
            projected = sm - rate * left if left is not None else None
            hopeless = left is not None and (
                left <= 0 or
                (self._n_robust >= self.patience
                 and projected > self.off_below + self.min_progress))
            if sm > self.off_below and hopeless:
                self._state, self._age = 0.0, -1
                self.handbacks += 1
                self.forced_handover = True
        else:
            self._robust_start, self._n_robust = None, 0
        self._age += 1
        self.history.append((len(self.history), s, self._state))
        return self._state


#: Default 3-stage ladder thresholds (envelope -> gc -> l2), from the gate
#: mining at s16: envelope drives skip to ~0.368 and gc to ~0.122, so hand over
#: envelope->gc at the 2-stage off_below (0.45) and gc->l2 at 0.20.
LADDER_THRESHOLDS = (0.45, 0.20)


class StagedMisfit(Misfit):
    """N-stage hard-switching objective: the >2-term generalisation of
    BlendedMisfit. Only the ACTIVE stage is ever evaluated (so extra stages are
    free), and each stage is divided by its OWN EMA adjoint-source norm, which
    keeps the gradient scale continuous across hand-overs -- the property that
    stops Adam/adagrad moment states seeing an orders-of-magnitude jump.

    Motivation (measured): weci = envelope->gc staged scores 0.451 at s16 while
    envelope alone is 0.240 and gc alone 0.210, and our 2-stage envelope->l2
    switch reaches 0.626. Since staging twice already beats either component,
    a 3-stage ladder envelope -> gc -> l2 (robust -> gentle refiner -> sharp
    refiner) is the natural extension. NB the counter-hypothesis is real: l2
    alone refines better than gc alone, so an intermediate gc stage may simply
    delay reaching l2 -- which is what the experiment settles.
    """

    def __init__(self, losses, names=None, beta=0.9, normalize=True, eps=1e-30,
                 allow_stateful=False):
        super().__init__()
        losses = list(losses)
        if len(losses) < 2:
            raise ValueError("StagedMisfit needs at least 2 stages")
        if not allow_stateful:
            for i, fn in enumerate(losses):
                _reject_stateful(f"stage{i}", fn)
        self.losses = losses
        self.names = ([str(n) for n in names] if names is not None
                      else [f"stage{i}" for i in range(len(losses))])
        if len(self.names) != len(losses):
            raise ValueError("names must match losses")
        self.beta, self.normalize, self.eps = beta, normalize, eps
        self._ema = {}
        self.stage = 0

    def set_stage(self, i):
        if not (0 <= int(i) < len(self.losses)):
            raise ValueError(f"stage must be in [0,{len(self.losses) - 1}], got {i}")
        self.stage = int(i)
        return self.stage

    @property
    def active_name(self):
        return self.names[self.stage]

    @property
    def scales(self):
        return dict(self._ema)

    def forward(self, a, b):
        wrt = BlendedMisfit._graph_arg(a, b)
        e = BlendedMisfit._eval(self.losses[self.stage], a, b)
        if not self.normalize:
            return e
        return e / _grad_scale(e, wrt, self._ema, self.stage, self.beta, self.eps)


class StageLadder:
    """Monotonic multi-stage controller driven by the measured skip fraction.

    Advances to the next stage when the EMA-smoothed skip falls below that
    stage's threshold, and NEVER goes back -- being a ladder rather than a
    switch, the hand-back/oscillation pathology cannot occur by construction
    (so no ratchet is needed). It may advance SEVERAL stages in one update: an
    easy start whose skip is already tiny should jump straight to the sharp
    refiner rather than waste iterations in robust mode ("stay out of the way").

    `thresholds` must be strictly descending, length n_stages - 1.
    """

    def __init__(self, thresholds=LADDER_THRESHOLDS, ema=0.5, dwell=1):
        th = [float(t) for t in thresholds]
        if not th:
            raise ValueError("need at least one threshold")
        if any(th[i] <= th[i + 1] for i in range(len(th) - 1)):
            raise ValueError(f"thresholds must be strictly descending, got {th}")
        self.thresholds = th
        self.ema, self.dwell = float(ema), int(dwell)
        self._smooth = None
        self.stage = 0
        self._age = self.dwell        # allow the FIRST update to advance
        self.history = []

    def update(self, skip):
        s = float(skip)
        if math.isfinite(s):
            self._smooth = s if self._smooth is None else \
                self.ema * s + (1.0 - self.ema) * self._smooth
            if self._age >= self.dwell:
                new = self.stage
                while new < len(self.thresholds) and self._smooth <= self.thresholds[new]:
                    new += 1
                if new != self.stage:
                    self.stage, self._age = new, -1
        self._age += 1
        self.history.append((len(self.history), s, self.stage))
        return self.stage


class DiagnosticLambda(LambdaSchedule):
    """PHASE B: lambda driven by the MEASURED cycle-skip fraction, with the
    frequency schedule as a prior and hysteresis so it cannot oscillate.

    Rationale: frequency is only a proxy for skip risk. A good starting model can
    carry L2 safely into high frequency, and a purely frequency-scheduled lambda
    would needlessly sacrifice L2's resolution there. This variant raises lambda
    only when skipping is actually observed.

    `on_above` / `off_below` are skip fractions; the gap between them is the
    hysteresis band (an observation inside it leaves lambda unchanged). Defaults
    are the Phase-1 gate calibration. `blend=1.0` = pure skip-driven (the
    verified configuration); lower it only to mix the frequency prior back in.
    For the single-scale staged switch use SkipSwitch instead.
    """

    def __init__(self, f_lo, f_hi, on_above=SKIP_ON_ABOVE, off_below=SKIP_OFF_BELOW,
                 stage_overrides=None, stage_anneal=1, blend=1.0):
        super().__init__(f_lo, f_hi, stage_overrides, stage_anneal)
        if not (0.0 <= off_below < on_above <= 1.0):
            raise ValueError("need 0 <= off_below < on_above <= 1")
        self.on_above, self.off_below, self.blend = on_above, off_below, float(blend)
        self._state = None           # committed weight; None -> init from 1st obs

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
        elif self._state is None:                  # first obs inside the band:
            self._state = 1.0                      # start robust (conservative)
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
