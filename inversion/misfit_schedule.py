"""Convex misfit scheduling (T6, spec E10).

    J(i) = sum_k w_k(i) * J_k,   with   sum_k w_k(i) = 1  for every i.

A ScheduledMisfit duck-types ADFWI's Misfit, so an instance drops into
AcousticFWI's ``loss_fn`` unchanged: AcousticFWI.calculate_loss calls
``loss_fn.forward(synthetic_waveform, observed_waveform)`` positionally, and
every component misfit here is called with exactly the same positional
argument order it would receive if used alone.

The wrapper OWNS the iteration counter (incremented once per forward call).
Never rely on a component misfit's internal counter: e.g. ADFWI's
``Misfit_weighted_ECI`` defaults to ``max_iter=1000``, which silently breaks
the sigmoid for short runs; here the horizon N is always explicit.

Counter reset policy (documented, deterministic):
- The counter starts at 0 (or ``start_iter``) and NEVER auto-resets.
- Across multiscale bands, ADFWI's convention is a GLOBAL iteration count
  (successive ``fwi.forward(iteration=..., start_iter=...)`` calls continue
  the count). If the schedule horizon N spans all bands, do nothing.
- If a band-local schedule is wanted (restart the ramp per band), call
  ``reset()`` (or ``reset(start_iter)``) explicitly when starting the band.

Component-misfit settings (spec T6; verified against the ADFWI sources):
- WECI-style two-term schedule: use ``weci_sigmoid_weights(N)`` with N = the
  PLANNED number of iterations. The code-verified form (Weci.py L38) is
  w_GC(i) = 1 / (1 + exp(-(i - N/2))); the SI of the paper misprints it.
- ``Misfit_sdtw(gamma, sparse_sampling, dt)``: use sparse_sampling >= 5 and
  gamma in {0.1, 1} (SoftDTW.py L29).
- ``Misfit_wasserstein_sinkhorn(dt, p, blur, scaling, sparse_sampling, ...)``:
  use dt = the TRUE propagator dt and sparse_sampling >= 5; blur/scaling are
  PROVISIONAL until tuned on synthetics (Wasserstein_sinkhorn.py L28).
"""

import math

from ADFWI.fwi.misfit import Misfit

_WEIGHT_TOL = 1e-9


def weci_sigmoid_weights(N):
    """Two-term WECI sigmoid schedule over a horizon of N planned iterations.

    Returns ``fn(i) -> (w_GC, w_other)`` with the CODE-verified form
    (ADFWI Weci.py L38):

        w_GC(i) = 1 / (1 + exp(-(i - N/2)))

    Component order convention: index 0 gets w_GC (the weight that grows from
    ~0 to ~1), index 1 gets 1 - w_GC (e.g. Envelope, dominant early).
    """
    N = float(N)

    def fn(i):
        w = 1.0 / (1.0 + math.exp(-(i - N / 2.0)))
        return (w, 1.0 - w)

    return fn


def linear_ramp_weights(N):
    """Two-term linear ramp: w0 goes 0 -> 1 linearly over N iterations.

    w0(i) = min(i / (N - 1), 1) for N > 1 (clamped at 1 beyond the horizon);
    N == 1 gives w0 = 1 immediately. Component order as in
    ``weci_sigmoid_weights``: index 0 ramps up, index 1 ramps down.
    """
    N = int(N)

    def fn(i):
        w = 1.0 if N <= 1 else min(i / (N - 1.0), 1.0)
        return (w, 1.0 - w)

    return fn


def piecewise_constant_weights(breakpoints, weight_rows):
    """Piecewise-constant weights per multiscale band.

    Args:
        breakpoints: increasing iteration indices ``[b_1, ..., b_M]`` at which
            the weights CHANGE; band m is ``b_m <= i < b_{m+1}`` (band 0 is
            ``i < b_1``).
        weight_rows: M+1 rows of weights (one per band, each summing to 1).

    Example (bands 5 -> 10 -> 20 Hz, three components):
        piecewise_constant_weights([30, 60], [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    """
    breakpoints = list(breakpoints)
    weight_rows = [tuple(float(w) for w in row) for row in weight_rows]
    if sorted(breakpoints) != breakpoints:
        raise ValueError("breakpoints must be increasing")
    if len(weight_rows) != len(breakpoints) + 1:
        raise ValueError("need exactly len(breakpoints) + 1 weight rows")
    K = len(weight_rows[0])
    for row in weight_rows:
        if len(row) != K:
            raise ValueError("all weight rows must have the same length")
        if abs(sum(row) - 1.0) > _WEIGHT_TOL:
            raise ValueError(f"weight row {row} does not sum to 1")

    def fn(i):
        band = 0
        for b in breakpoints:
            if i >= b:
                band += 1
            else:
                break
        return weight_rows[band]

    return fn


class ScheduledMisfit(Misfit):
    """Convex combination of component misfits with iteration-driven weights.

    Args:
        components: list of ADFWI Misfit instances (the J_k). Each is called
            positionally as ``component.forward(syn, obs)`` with the same
            argument order AcousticFWI uses.
        weight_fn: callable ``i -> sequence of len(components) weights``.
            Weights must sum to 1 at every i (validated on every call).
        start_iter: initial value of the owned iteration counter. Default 0.

    The counter increments by exactly 1 per ``forward`` call and never
    auto-resets; see the module docstring for the multiscale reset policy.
    """

    def __init__(self, components, weight_fn, start_iter=0):
        super().__init__()
        if len(components) < 1:
            raise ValueError("need at least one component misfit")
        self.components = list(components)
        self.weight_fn = weight_fn
        self.iter = int(start_iter)
        # validate the schedule at the starting iteration
        self._weights(self.iter)

    def _weights(self, i):
        w = tuple(float(x) for x in self.weight_fn(i))
        if len(w) != len(self.components):
            raise ValueError(
                f"weight_fn returned {len(w)} weights for "
                f"{len(self.components)} components")
        if abs(sum(w) - 1.0) > _WEIGHT_TOL:
            raise ValueError(f"weights {w} at iteration {i} do not sum to 1")
        return w

    def forward(self, syn, obs):
        w = self._weights(self.iter)
        loss = None
        for wk, comp in zip(w, self.components):
            term = wk * comp.forward(syn, obs)
            loss = term if loss is None else loss + term
        self.iter += 1
        return loss

    def reset(self, start_iter=0):
        """Reset the owned iteration counter (e.g. for a band-local schedule
        when a new multiscale band starts). Explicit by design - never called
        automatically."""
        self.iter = int(start_iter)
