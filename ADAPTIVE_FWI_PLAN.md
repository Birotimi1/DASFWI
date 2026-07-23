# Adaptive Frequency-Continuous DAS-FWI — Research Plan

**Status (2026-07-23):** Marmousi elastic A/B campaign running (Phase 0). When it
finishes we begin **Phase 1: the cycle-skipping flip test** — the hypothesis gate
for everything below. This plan was designed in dialogue (Opus) and verified
mathematically (Fable); Fable's four amendments are folded in.

---

## 0. Purpose and scientific premise

**Product:** recover **Vp and Vs** from DAS **strain rate** alone, with **no
auxiliary field data** (no sonic logs, no check-shots) — an *exploratory-phase*
capability, deployable anywhere. **Transferability is the deliverable**, which is
why every choice below avoids region-calibrated priors and externally-trained
components.

**Observable:** E3 gauge strain rate, ε̇ = [v_z(z+l/2) − v_z(z−l/2)] / l, consumed
directly (NO velocity conversion; autograd builds the adjoint through the layer).

**Central hypothesis (to be TESTED, not assumed):** with a good starting model
and low frequency, plain **L2 wins** (max-likelihood, highest resolution); as
frequency climbs and cycle-skipping sets in, **Wasserstein–Sinkhorn OT (and the
robust misfit family) overtake L2**. Confirmed so far ONLY in the no-skip regime
(acoustic Marmousi: `l2_adam` best, SSIM 0.868). The flip is unverified on strain
rate — **Phase 1 exists to confirm or refute it.**

**End-state pipeline (what we are building toward):**

```
DAS strain rate
   └─ Route B: wave-equation cross-correlation traveltime  → Vp starting model
   └─ Vs seed = Vp/√3 (physics prior)                       → refined by S kinematics
        └─ Sequential elastic FWI (Vp-lead, Vs-follow)
             └─ multiscale low→high frequency
                  └─ adaptive misfit  λ(f, stage): L2 ──▶ Sinkhorn/OT
                       → Vp, Vs models
```

---

## 1. Verified premises (Fable's checks — quantitative)

1. **Cycle-skip criterion:** onset when kinematic misalignment |Δt| > T/2 =
   1/(2·f_max). Frequency is therefore a *risk proxy*; the true driver is Δt·f, so
   a good starting model can legitimately carry L2 into high frequency. (This is
   why λ should ultimately be diagnostic-primed, not purely frequency-scheduled.)
2. **E3 preserves first-arrival kinematics.** Onset bias ≤ (l/2)/c: ≤ ~13–27 ms
   for the 80 m Marmousi gauge, ~1–3 ms for FORGE's 10 m gauge — negligible vs
   T/2 = 100–200 ms at band 1. **Stronger point:** in Route B the E3 operator is
   applied to the SYNTHETICS too, so syn and obs are shaped identically and the
   cross-correlation time shift carries **zero operator-induced kinematic bias**.
   This is the single strongest argument that the pipeline is DAS-native and
   transferable — state it in code docstrings.
3. **OT convexity is a hypothesis on oscillatory strain rate, not a theorem.**
   Wasserstein convexity-in-shift holds for non-negative mass-normalized signals;
   seismic data inherits it only through positivity transforms + careful
   normalization. **Standing warning:** NIM (a W1-type transport misfit) DIVERGED
   on strain rate under all 5 optimizers. So "OT leads under skipping" must be
   demonstrated, not presumed.
4. **√3 Vs-seed caveat (design-changing).** Vp/Vs = √3 ⇔ ν = 0.25, correct for
   crystalline basement (1.70–1.75). BUT ~1 km of sedimentary cover with true
   Vp/Vs ≈ 2.2 seeded at 1.73 gives a one-way S delay error ~175–210 ms — **at or
   beyond T/2 = 167 ms at 3 Hz.** The S wavefield can cycle-skip *at the starting
   frequency* in the cover. Fix: **λ is per-(f, stage)** — the Vs-release stage
   starts at λ=1 regardless of band, then anneals.
5. **DAS S-sensitivity ~ sinθ·cosθ** (zero along-axis and broadside, max at 45°):
   Vs illumination is offset-dependent. Do not assume uniform shot weighting in
   the Vs stage; log it when interpreting Vs recovery.

---

## 2. Locked design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Route B** (wave-equation cross-correlation traveltime) for the starting model; **FNO deferred** | picking-free, DAS-native, zero operator bias (§1.2); FNO is garbage-in-garbage-out on picks, distribution-dependent (anti-transferability), and unnecessary once Route B removes the eikonal |
| D2 | **Vs seed = Vp/√3** (physics prior), NOT Castagna; refine via S kinematics inside the first elastic stage | √3 = universal isotropic default; Castagna is a clastic-basin fit → not transferable. Explicit up-front S-tomography is redundant with the convex low-freq elastic stage |
| D3 | **Sequential Vp-lead, Vs-follow with overlap** | suppresses the Vp–Vs cross-talk that made the joint 3-parameter run diverge; overlap stops Vp absorbing S kinematics |
| D4 | **Adaptive misfit λ(f, stage): L2 → Sinkhorn**, continuous ramp | L2 for resolution where safe, OT for robustness where skipping; continuous keeps the objective/gradient smooth |
| D5 | **Blend = normalized + short-circuited**: (1−λ)·E_L2/s_L2 + λ·E_OT/s_OT, s_i = detached EMA; never evaluate the zero-weighted term | L2 and OT differ ~orders of magnitude; naive sum makes λ meaningless. Sinkhorn ~20× L2 cost → short-circuit at λ∈{0,1} |
| D6 | **λ driver:** frequency-scheduled first; **diagnostic-primed** upgrade later | schedule is simple/deterministic; diagnostic preserves L2 resolution when a good start keeps you aligned at high f (§1.1) |
| D7 | **Density held constant** (2450) unless a dedicated multi-param study | joint ρ diverged and dragged Vp/Vs (finding #5); revisit only with parameter-scaling |
| D8 | **All rankings dual** — SSIM/MAPE (structure) + RMS/dRMS (amplitude), separate tables | they disagree (l2_nadam vs l2_adam); Liu's metrics (SSIM Wang 2004, MAPE Hyndman & Koehler 2006) |

---

## 3. Standing engineering rules (every phase)

- Develop in the **local clone** (`scratchpad/DASFWI_work`); the SMB mount
  (`/Volumes/AS-Filer/...`) has broken writes — edit + commit + push from the
  clone, user pulls on OrangeGrid.
- **`py_compile` + a fabricated-data functional test** for every script BEFORE
  push. New shell scripts get exec bits via `git update-index --chmod=+x`.
- Every misfit / technique registers in **`inversion/config.py`** (single source
  of truth). No copy-paste of technique definitions.
- **Never `rm results/` while jobs run** (`condor_rm` first). Runners re-mkdir
  their out_dir before saving as a backstop.
- OrangeGrid submit conventions: `+request_gpus = 1`,
  `Requirements = (CUDADriverVersion >= 12.0) && (CUDACapability >= 8.0)`,
  conda via `$HOME/miniconda3` hook, `output/`+`logs/`. Scope health checks to the
  live cluster (`status.sh`).
- Commits: `Co-Authored-By: Birotimi <Birotimi@syr.edu>` only.

---

## 4. Phased plan — each phase is a falsifiable gate

### Phase 0 — Close the elastic baseline (no new code)
1. Wait for 90/90 elastic A/B; `rank_campaign.py --csv`; rsync results + figures local.
2. Archive the locked baseline: acoustic dual table, elastic dual table, illumination
   A/B conclusion (expected: helps SGD at depth, wash for adam-family).
- **Acceptance:** both CSVs archived; findings memory updated.

### Phase 1 — Cycle-skip flip test  ← THE HYPOTHESIS GATE (start here after Phase 0)
3. **`inversion/skip_diagnostic.py`** — per-trace windowed cross-correlation lag
   between syn and obs; `skip_fraction = mean(|lag| > 1/(2·f_max))`. Torch,
   batched, detached, cheap; log every iteration. Unit test: shifted-Ricker traces
   with a known analytic skip fraction.
4. **Band-filter utility** — verify/​wrap `fwi/multiScaleProcessing.py`; zero-phase
   low/high-pass applied IDENTICALLY to syn and obs. Unit test on spectra.
5. **Induce skipping — FULL grid (user decision, 2026-07-23).** Do NOT pre-trim:
   run the complete **45-combo grid (9 misfits × 5 optimizers, INCLUDING nim** —
   its behaviour *under skipping* is a data point, not a reason to omit), same
   completeness as the 90-job campaign, scored with the full **dual ranking
   (SSIM/MAPE + RMS/dRMS)**. Rationale: the optimizer×misfit ordering can reorder
   under skipping, so assuming the no-skip winners (adam) transfer would defeat the
   test's purpose.
   - **Platform: ACOUSTIC Marmousi (Vp-only).** Cheap enough to afford 45×rungs,
     and isolates the misfit×skip physics without the Vp/Vs/density/staging
     confounds. (The flip is a fundamental misfit property; confirm it carries to
     elastic later, in Phase 4.) The finished acoustic campaign is the no-skip
     reference rung.
   - **Primary induction axis: starting-model degradation LADDER** (extend
     `inversion/run_starting_model_ladder.py` to the full optimizer grid + dual
     metrics + skip-diagnostic logging). ~4–5 rungs good→bad, e.g. Gaussian
     σ ∈ {6(≈current/good), 12, 24, 48 nodes} + a data-independent **1-D linear
     v(z)** worst rung.
   - **Cost:** 45 × ~4–5 rungs ≈ 180–225 acoustic jobs; the slow misfits
     (sinkhorn ~8h, sdtw, convsi ~5h) are the burn — the fast ones are ~0.4h.
     Rung count is the tunable dial if wall-clock is tight (keep all 45 combos).
   - **Secondary axis (Phase 1b, only if needed):** low-frequency deprivation,
     high-pass obs at f_hp ∈ {2,4} Hz — stresses a different part of the objective.
6. Dual-rank **each rung** → the **flip curve** (SSIM AND dRMS vs starting-model
   rung, per misfit×optimizer) + per-iteration skip-fraction traces.
- **Acceptance:** L2 SSIM degrades monotonically with rung; the **flip rung**
  (a robust misfit overtakes L2 in SSIM) is identified — or shown absent; the
  logged skip fraction at band start correlates with L2 failure (record the
  EMPIRICAL threshold; do not assume one). **If no flip → STOP, redesign, build
  nothing further.**

### Phase 2 — Adaptive λ objective (only after a confirmed flip)
7. **`inversion/adaptive_misfit.py`** — `BlendedMisfit(loss_lo, loss_hi, lam)`:
   normalized short-circuited blend (D5); schedule
   `λ_b = clip((ln f_b − ln f_lo)/(ln f_hi − ln f_lo), 0, 1)` with (f_lo,f_hi) from
   the Phase-1 flip point; **per-stage override table** (√3 amendment). Register as
   `adaptive` in `config.py`. Unit tests: λ=0/1 reduce EXACTLY to the pure misfits;
   finite gradient at λ=0.5 on strain-rate-scale (~1e-8) data.
8. **`inversion/run_adaptive.py`** (acoustic first) — configurable bands
   (e.g. 2.5→5→7.5→full Hz), per-band iteration budget, symmetric filtering,
   skip-fraction logged.
9. Validate at the flip rung — three arms: fixed L2, fixed sinkhorn, adaptive.
- **Acceptance:** adaptive ≥ best fixed arm in final SSIM (tol ~0.01) AND
  final-band MAPE at L2-grade AND not worse than L2 anywhere below the flip band.

### Phase 3 — Route B starting model (transferable initial)
10. **`inversion/run_traveltime_starter.py`** — from a data-independent 1-D linear
    v(z); traveltime misfit at the lowest band; heavy smoothness (GradProcessor
    smoothing + Tikhonov-2); ~50–100 iters → `vp_start`.
11. **Acceptance (quantitative, from Phase 1):** L2 skip fraction at band 1 under
    `vp_start` below the empirical flip threshold (target < ~10%); adaptive FWI from
    `vp_start` within ε of the 180 m-smooth-start reference SSIM.
12. **Vs seed:** `vs_start = vp_start/√3`, config hook for a depth-graded ratio
    table (cover vs basement); Vs-release stage forced λ=1 initially (§1.4).
    Document the cover-skip caveat in the docstring.

### Phase 4 — Integration
13. Elastic adaptive driver, 2-D schedule (band × parameter stage): band 1 Vp-only;
    Vs released band 2+ with λ_vs annealing; Poisson clamp retained; illumination
    precond per Phase-0 (on for sgd, off for adam-family).
14. **Full-pipeline Marmousi test:** 1-D start → Route B → adaptive elastic →
    dual metrics vs truth.
- **Acceptance:** decisively beats fixed-L2-from-1-D-start; approaches the
  smoothed-start baseline.
15. **FORGE staging:** field loader + `convsi` as the hi-λ term (unknown source —
    decide OT vs convsi for FIELD; test both on one shot line first).

### Phase 5 — Optional upgrades (only if earned)
16. Diagnostic-driven λ WITH hysteresis (no oscillation) if Phase-2 logs show the
    schedule over-invoking OT. Eikonal-FNO only if tomography speed is a bottleneck.

---

## 5. Immediate next action

**When 90/90 elastic finishes:** Phase 0 close-out, then **Phase 1** — build
`skip_diagnostic.py` + the band filter, induce skipping on acoustic Marmousi, and
produce the **flip curve**. That single result decides whether Phases 2–5 proceed
as written or the objective design is reconsidered.

> The discipline that matters most: Phase 1 is a hypothesis test about strain-rate
> objectives (NIM's divergence is the standing reminder that OT-family behavior on
> this observable is not free). Nothing downstream is built until the flip curve exists.
