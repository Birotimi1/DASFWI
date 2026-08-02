# Adaptive Frequency-Continuous DAS-FWI — Research Plan

**Status (2026-07-23):** Marmousi elastic A/B campaign running (Phase 0). When it
finishes we begin **Phase 1: the cycle-skipping flip test** — the hypothesis gate
for everything below. This plan was designed in dialogue (Opus) and verified
mathematically (Fable); Fable's four amendments are folded in.

---

# ⚑ THE GOVERNING PLAN (2026-07-31) — FOLLOW IN ORDER, DO NOT DEVIATE

**Every starting model comes from Route B — wave-equation cross-correlation
traveltime, NO picking** (`run_traveltime_starter.py`). That is what we will have
at FORGE, so it is what the validation must use. Smoothed-truth starts leak the
answer and are **not** acceptable as validation, even though the rung results
already proved the switch works in that controlled setting.

**The regime is chosen by FREQUENCY, not by degrading the model.** With a fixed
start, skipping is |Δt| > T/2 = 1/(2·f_max): low band → non-skip, high band →
skip. `skip_vs_band()` returns the whole curve from ONE forward (the lags do not
depend on the band, only the threshold does), and the starter prints it labelled
NO-SKIP / transition / SKIP. **Pick the test bands from that table, not by
guessing.**

| # | step | the question it answers | acceptance |
|---|------|------------------------|------------|
| 1 | **acoustic Route B starter** | can wave-equation xcorr build a usable start? | `skip@starter` < `skip@1-D` |
| 2 | acoustic, **NON-SKIP** band | do we recover the true model? does the switch stay out of L2's way? | switch ≥ L2, both recover |
| 3 | acoustic, **SKIP** band | does L2 fail, and does the switch rescue it? | switch > L2 |
| 4 | acoustic **multiscale + switch combos** | both regimes | — |
| 5 | **elastic**: repeat 1–4 | does it carry to Vp+Vs? | — |
| 6 | **FORGE field** | the real thing | — |

**Fallback:** if Route B cannot produce a usable start, do **not** patch around
it — move to an **eikonal solver**. Park et al. (2025) use manual picks + a
hybrid eikonal solver at this exact site.

**Settled — do not re-run:** the acoustic switch on smoothed-truth rungs (60
cells, s6/s16/s20, 5 optimizers — switch wins everywhere); acoustic multiscale
(NEGATIVE, the cascade hurts); the elastic regression (0.583/0.702, reproduces
the campaign — code validated).

**Before any HPC submission:** `--dry-run` (config) **then** `--smoke`
(execution). Both are required — `--dry-run` exits before the iteration loop and
cannot catch runtime errors.

---

# 📁 CODEBASE MAP (for a fresh reader or a model picking this up cold)

```
DASFWI/
├── ADFWI_local/          Liu Feng's ADFWI, vendored. Propagators, misfits,
│                         optimizers, the FWI loops. `ADFWI.fwi.AcousticFWI`
│                         auto-routes LBFGS/NLCG through forward_closure().
├── inversion/            OUR method code (the science)
│   ├── config.py             SINGLE SOURCE OF TRUTH: MISFITS, LIU_OPTIMIZERS,
│   │                         MISFIT_SETTINGS (batch/checkpoint/normalize).
│   │                         Every driver imports from here.
│   ├── adaptive_misfit.py    THE SWITCH. BlendedMisfit (grad-norm normalised,
│   │                         short-circuited), SkipSwitch (binary lambda +
│   │                         hysteresis + dwell + ratchet + stall guard),
│   │                         StagedMisfit/StageLadder (N-stage), LambdaSchedule,
│   │                         DiagnosticLambda, stage_plan.
│   ├── skip_diagnostic.py    THE TRIGGER. trace_lags (FFT xcorr), skip_fraction,
│   │                         skip_vs_band (whole curve from ONE measurement),
│   │                         ricker_f90, skip_threshold.
│   ├── das_conditioning.py   Noe et al. conditioning: wavelength_span (lambda/4),
│   │                         arrival_window, channel_weights, ConditionedMisfit.
│   ├── starting_model.py     linear_vz, vs_from_vp (sqrt3), poisson_clamp.
│   ├── safe_misfits.py       numerics-hardened misfit subclasses + apply_misfit.
│   └── metrics.py            SSIM + MAPE (Liu's metrics).
├── forge/                FORGE field: proxy_model, traveltime_tomography
│                         (STA/LTA picking -- the eikonal-style fallback path).
├── das/                  DAS observation operator (E3 gauge -> strain rate).
├── hpc/
│   ├── marmousi_full_das/    ACOUSTIC campaign + the plan's steps 1-4
│   │   ├── common.py             grid/source/acquisition, load_models(rung),
│   │   │                         START_RUNGS, build_* helpers. 88x200 @ 40 m.
│   │   ├── run_one.py            one campaign cell (the 90-job gate)
│   │   ├── run_traveltime_starter.py  STEP 1: Route B starter -> starter/<tag>/
│   │   ├── run_switch.py         STEPS 2-3: the switch experiment
│   │   ├── run_adaptive.py       STEP 4: multiscale
│   │   ├── calibrate_rungs.py, flip_curve.py, mine_gate.py, handover_sweep.py
│   │   └── rank_switch.py / rank_adaptive.py / rank_campaign.py   READERS
│   ├── elastic_full_das/     ELASTIC (78x200 @ 45 m, F0=3 Hz) -- STEP 5
│   │   ├── run_one.py, run_pipeline.py, run_traveltime_starter.py
│   ├── standalone/           single runs incl. run_field_das.py (FORGE)
│   ├── condor/               OrangeGrid (HTCondor). run_standalone.sh maps
│   │                         kind -> script and is SCHEDULER-AGNOSTIC.
│   ├── slurm/                Bridges-2. submit.sh / submit_array.sh /
│   │                         activate_bridges2.sh (conda; DASFWI_ACTIVATE).
│   └── check_progress.py     live progress + starter matrix verdict
├── tests/                130 tests. conftest.py puts ADFWI on sys.path.
└── results/              NOT in git. <campaign>/<tag>/{iter_vp,iter_loss,
                          metrics.json,final.png}
```

**Conventions that matter if you touch this code:**
- **Every driver must checkpoint** (~25 iterations) — a walltime kill otherwise
  loses the whole cell. Sweep them all before any launch.
- **Every knob that changes an experiment must be in the output tag**, or runs
  overwrite each other. This class of bug has appeared four times.
- **`--dry-run` validates config, `--smoke` validates execution.** Both are
  required before submitting; dry-run exits before the loop.
- Results are read by `rank_switch.py --rung {s6,s16,s20,routeb}`,
  `rank_adaptive.py [--elastic]`, `check_progress.py`.

---

## ⛔ PHASE-1 GATE RESULT (2026-07-29, Bridges-2, 90 cells) — READ FIRST

**The L2→OT hypothesis below is REFUTED; the adaptive objective retargets to
L2↔ENVELOPE.** Mean SSIM over the 5 optimizers, by starting-model rung
(measured start skip fraction in parens):

| misfit | s6 (0.51) | s16 (0.64) | s20 (0.79) |
|---|---|---|---|
| l2 | **0.812** | 0.326 | 0.261 |
| weci | 0.759 | **0.451** | **0.341** |
| sinkhorn (OT) | 0.759 | 0.245 | 0.206 |

`flip_curve.py` verdict: **FLIP AT s16** — L2 wins at 51% skip, collapses by 64%;
the envelope-family (weci) wins from there, degrading gracefully. **Sinkhorn is
never above L2 and craters with it** — OT is not the cycle-skip cure on DAS
strain rate; the classical phase-insensitive misfits are. Also an **optimizer
flip**: adagrad, worst at low skip, is the best optimizer under strong skip.

**Fable verification of the redesign (REVISE, folded in):**
- `weci` is itself a hardcoded staged switch (envelope for iters 0–150, then a
  sigmoid hand-over to global correlation) and is **stateful** — it cannot sit
  inside a switching controller. The robust term is stateless
  **`Misfit_envelope`**; the refinement half of weci's schedule is exactly what
  our L2 leg supplies, with skip-driven (not iteration-hardcoded) timing.
- Thresholds: **on_above=0.58, off_below=0.45** (the originally proposed 0.65
  sat above s16's measured 0.64 and would never have fired). off_below to be
  refined empirically by the hand-over sweep. Hand-back is ratcheted; re-entries
  are logged as an abort signal.
- Phase A runs **single-scale, full band, at s16** (the gate's exact calibration
  conditions; s20 secondary, s6 sanity). Multiscale = Phase B, after per-band
  threshold re-verification. Elastic validation before any field use.

**Implemented (inversion/adaptive_misfit.py + hpc/marmousi_full_das/):**
`SkipSwitch` (binary λ, init-from-first-measurement, EMA, dwell, ratchet),
`mine_gate.py` (free calibration from the gate's metrics), `handover_sweep.py`
(~2 SU: probe cached snapshots + short L2 restarts → empirical off_below),
`run_switch.py` (Phase A arms: switch / fixedk / l2 / envelope, checkpointed,
logs the skip/λ trajectory), `rank_switch.py` (verdict vs the gate controls).

**Phase A run order (Bridges-2):**
```bash
python hpc/marmousi_full_das/mine_gate.py                     # free, login node
hpc/slurm/submit.sh handover                                  # ~2 SU, 1 GPU job
# read both; adjust --off-below if the sweep says so, then (~10 SU):
for OPT in adam adagrad sgd; do
  hpc/slurm/submit.sh switch -- --arm switch --start-rung s16 --optimizer $OPT
  hpc/slurm/submit.sh switch -- --arm fixedk --start-rung s16 --optimizer $OPT
done
python hpc/marmousi_full_das/rank_switch.py --rung s16        # the verdict
# WIN = switch >= weci + 0.05 SSIM and >= l2 (and beats fixedk to justify the
# diagnostic). Controls (l2/envelope/weci x 5 opts) come from the gate — free.
```

Everything below this line is the original plan, kept for the record; read the
OT-based Phases 2–5 through the lens of this result (`sinkhorn` → `envelope`).

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
| D5 | **Blend = GRADIENT-NORM-normalized + short-circuited**: (1−λ)·E_L2/s_L2 + λ·E_OT/s_OT with **s_i = detached EMA of ‖∂E_i/∂m‖** (NOT of the loss value) | **Measured on real strain-rate-scale data (2.3e-8):** L2 value 5.6e-7 / ‖g‖ 0.49; sinkhorn 1.3e-4 / 1.3e3; gc **−5.2e-2** / 2.5e5. Value ratio (sinkhorn/L2) = 233 but GRADIENT ratio = 2686 — they differ **11.5×**, so value-normalization does not equalize influence on the update; and **gc is negative**, which breaks value-normalization outright. Get per-term grads via `torch.autograd.grad(..., retain_graph=True)` (forward graph shared; refresh the EMA every K iters to amortize). Sinkhorn ~20× L2 cost → short-circuit at λ∈{0,1} |
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

> **IMPLEMENTED 2026-07-24.** Three corrections were found while verifying this
> phase against the code; they are folded in below and were the reason not to
> implement it as originally specified.
>
> **C1 — build on the CAMPAIGN infrastructure, not `run_starting_model_ladder.py`.**
> That file is a *different experiment*: a 201x301 @ 5 m crop, f0 = 10 Hz, 4 shots,
> 1 fiber, hardcoded `sgd/lr=0.004`, via `run_inverse_crime`. Our campaign is
> 88x200 @ 40 m, 40 shots, 4 fibers, f0 = 5 Hz. Using it would make rung 0
> incomparable to the baseline. Instead `hpc/marmousi_full_das/run_one.py` gained
> `--start-rung`, so **rung s6 IS the finished 45-combo campaign (free)** and every
> rung **reuses the same `obs_data_das.npz`** (observed data depends only on
> `vp_true`) — nothing to regenerate.
>
> **C2 — the rung ladder must concentrate on sigma 12..24; sigma >= 32 is wasted.**
> Vertical-traveltime analysis on a Marmousi-like proxy vs the T/2 threshold:
> s6 ~16 ms (0% skipped) | s12 ~43 ms (0%) | s16 ~65 ms (**4%**) | s24 ~112 ms
> (**96%**) | s32 ~153 ms (100%) | s48 ~206 ms (100%). The transition is SHARP
> between s16 and s24, so the original {6,12,24,48} would have spent three rungs in
> the same saturated regime with no sampling inside the transition. Also
> counterintuitive: **1-D linear v(z) is NOT the worst rung** kinematically (~47 ms,
> 0% skipped) — it preserves the average v(z). *Ordering by RMS != ordering by skip
> risk.*
>
> **C3 — the skip threshold is ~80 ms, not 100 ms.** The source is an INTEGRATED
> Ricker: measured f_peak = 3.54 Hz (not 5), f90 = 6.25 Hz -> T/2 = 80 ms. Use the
> band's f90 (`skip_diagnostic.ricker_f90`), never the nominal f0.

3. **`inversion/skip_diagnostic.py`** ✅ — FFT cross-correlation per-trace lag on
   (n_shots, nt, n_chan); `skip_fraction = frac(|lag| > 1/(2·f_max))`; dead-trace
   guard (DAS blind channels); returns lag stats + peak correlation. Positive lag =
   synthetic arrives LATE. Torch, batched, detached. `tests/test_skip_diagnostic.py`
   (6 tests: known shifts, threshold, dead traces, amplitude invariance, f90).
   Also wired into `run_one.py`: `skip_init` / `skip_final` land in `metrics.json`.
4. **Band-filter utility** — for the OPTIONAL Phase-1b low-frequency-deprivation
   axis. NOTE `fwi/multiScaleProcessing.py` provides **`lowpass` only — there is no
   highpass**, so that axis needs a new filter (not free, as previously assumed).
   Deferred until 1a/1 results say whether it is needed.
5. **Phase 1a — RUNG CALIBRATION FIRST** ✅ `hpc/marmousi_full_das/calibrate_rungs.py`.
   One **forward only** (no inversion) per candidate rung against the shared obs →
   skip fraction per rung, ~1–2 min each. Pick the rungs it labels **TRANSITION**
   (5%–85% skipped); rungs at 0% or 100% teach nothing and cost 45 jobs each.
   Submit: `condor_submit hpc/condor/run.sub -a 'kind=calibrate'`.
6. **Phase 1 — FULL grid on the calibrated rungs (user decision, 2026-07-23).**
   Do NOT pre-trim: the complete **45-combo grid (9 misfits × 5 optimizers,
   INCLUDING nim** — its behaviour *under skipping* is a data point), scored with
   the full **dual ranking (SSIM/MAPE + RMS/dRMS)**. Rationale: the
   optimizer×misfit ordering can reorder under skipping, so assuming the no-skip
   winners (adam) transfer would defeat the test's purpose.
   - **Platform: ACOUSTIC Marmousi (Vp-only)** — affordable at 45×rungs and it
     isolates the misfit×skip physics from Vp/Vs/density/staging confounds. (The
     flip is a fundamental misfit property; confirm it carries to elastic in
     Phase 4.) The finished campaign is rung s6, the no-skip reference.
   - **Rungs:** `START_RUNGS = s6, s12, s16, s20, s24, const` (s6 free). Generate
     the job list with `make_ladder_combos.sh s16 s20 s24 ...` (refuses s6).
   - **Cost:** 45 × n_rungs. 3 calibrated rungs ≈ 135 jobs; the slow misfits
     (sinkhorn ~8 h, sdtw/convsi ~5 h) dominate. **Rung count is the dial** if
     wall-clock is tight — keep all 45 combos.
   - Submit: `condor_submit hpc/condor/skip_ladder.sub`.
   - **Optional Phase 1b (only if 1a/1 demand it):** low-frequency deprivation —
     needs a highpass that `multiScaleProcessing` does not provide (see step 4).
7. **`flip_curve.py`** ✅ — per-rung leaderboards in BOTH metric families, the
   misfit×rung SSIM curve, and an explicit **VERDICT**: the flip rung, or
   "NO FLIP → stop and redesign". Verdict logic verified on fabricated flip and
   no-flip scenarios. `--csv`, `--plot`.
- **Acceptance:** L2 SSIM degrades monotonically with rung; the **flip rung**
  (a robust misfit overtakes L2 in SSIM) is identified — or shown absent; the
  logged skip fraction at band start correlates with L2 failure (record the
  EMPIRICAL threshold; do not assume one). **If no flip → STOP, redesign, build
  nothing further.**

### Phase 2 — Adaptive λ objective  ✅ BUILT 2026-07-24 (validation still gated on Phase 1)

> The MACHINERY is parameterized, so it is built and unit-tested now; what Phase 1
> supplies is the NUMBERS (`--flip-lo/--flip-hi`) and the acceptance verdict. The
> defaults in the code are placeholders, **not physics**.
>
> - `inversion/adaptive_misfit.py` — `BlendedMisfit` (adjoint-source
>   gradient-norm normalisation, EMA scales, short-circuit at λ∈{0,1}, NIM-safe
>   via `apply_misfit`), `LambdaSchedule` (log-linear f-ramp + per-stage
>   overrides), `DiagnosticLambda` (**Phase 5**, skip-fraction-driven with
>   hysteresis), `stage_plan` (the 2-D band × stage resolver).
> - `hpc/marmousi_full_das/run_adaptive.py` — acoustic multiscale driver with the
>   three arms (`--objective adaptive|l2|sinkhorn`) for the acceptance test.
> - `tests/test_adaptive_misfit.py` — 14 tests, incl. the central claim that
>   normalisation gives each term a unit-norm adjoint source, the guard test that
>   the raw terms differ by >100×, endpoint direction-equivalence, and that the
>   expensive OT term is never evaluated at λ=0.
7. **`inversion/adaptive_misfit.py`** — `BlendedMisfit(loss_lo, loss_hi, lam)`:
   **gradient-norm**-normalized, short-circuited blend (D5 — do NOT normalize by
   loss value; verified 11.5× discrepancy and gc is sign-indefinite); schedule
   `λ_b = clip((ln f_b − ln f_lo)/(ln f_hi − ln f_lo), 0, 1)` with (f_lo,f_hi) from
   the Phase-1 flip point; **per-stage override table** (√3 amendment). Register as
   `adaptive` in `config.py`. Unit tests: λ=0/1 reduce EXACTLY to the pure misfits;
   finite gradient at λ=0.5 on strain-rate-scale (~1e-8) data; and the two terms'
   post-normalization gradient norms match to within a small tolerance.
8. **`inversion/run_adaptive.py`** (acoustic first) — configurable bands
   (e.g. 2.5→5→7.5→full Hz), per-band iteration budget, symmetric filtering,
   skip-fraction logged.
9. Validate at the flip rung — three arms: fixed L2, fixed sinkhorn, adaptive.
- **Acceptance:** adaptive ≥ best fixed arm in final SSIM (tol ~0.01) AND
  final-band MAPE at L2-grade AND not worse than L2 anywhere below the flip band.

### Phase 3 — Route B starting model  ✅ BUILT 2026-07-24

> - `inversion/starting_model.py` — `linear_vz` (data-independent 1-D start),
>   `vs_from_vp` (√3 Poisson-solid prior + optional depth-graded ratio for
>   cover/basement), `poisson_clamp`, `smooth_model`. 7 tests, incl. that √3
>   implies ν = 0.25 exactly and that the depth-graded seed switches at the
>   configured contact depth.
> - `hpc/marmousi_full_das/run_traveltime_starter.py` — 1-D linear start →
>   cross-correlation traveltime at the lowest band with Tikhonov-2 smoothness →
>   `starter/vp_start.npz` (+ Vs seed). Prints the skip fraction of BOTH the 1-D
>   start and the delivered starter, which is the acceptance number.
10. **`inversion/run_traveltime_starter.py`** — from a data-independent 1-D linear
    v(z); traveltime misfit at the lowest band; heavy smoothness (GradProcessor
    smoothing + Tikhonov-2); ~50–100 iters → `vp_start`.
11. **Acceptance (quantitative, from Phase 1):** L2 skip fraction at band 1 under
    `vp_start` below the empirical flip threshold (target < ~10%); adaptive FWI from
    `vp_start` within ε of the 180 m-smooth-start reference SSIM.
12. **Vs seed:** `vs_start = vp_start/√3`, config hook for a depth-graded ratio
    table (cover vs basement); Vs-release stage forced λ=1 initially (§1.4).
    Document the cover-skip caveat in the docstring.

### Phase 4 — Integration  ✅ BUILT 2026-07-24

> `hpc/elastic_full_das/run_pipeline.py` — the whole proposal end to end:
> `--start linear|route_b|smooth` → multiscale cascade → λ(f, stage) blend →
> 2-D staging (band 1 Vp-only; Vs released at `--vs-release-band` with λ forced
> to `--vs-lambda-start` then annealed) → Poisson clamp, water mask, optional
> illumination precond. `--fixed <misfit>` gives the control arms. Density stays
> constant. Schedule resolved by the unit-tested `stage_plan`.
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

## 5. Running the pipeline (all phases BUILT 2026-07-24)

Everything is implemented and unit-tested (**105 tests pass**). The science code
is scheduler-agnostic; the scheduler layer is thin and now exists for **both**
clusters:
- **OrangeGrid (HTCondor):** `hpc/condor/` — see its `README.md`.
- **PSC Bridges-2 (SLURM + H100):** `hpc/slurm/` — see its `README.md`. Same job
  "kinds" and combos, reusing the condor wrappers via `DASFWI_ACTIVATE`. Budget:
  allocation `ees260010p`, **1,657 SU ≈ 828 H100-hr** (H100 = 2 SU/hr), so
  **calibrate one run and throttle the array** before the full sweep.

The commands below are shown for OrangeGrid; the `hpc/slurm/submit.sh` /
`submit_array.sh` equivalents are one-to-one (mapping in `hpc/slurm/README.md`).

**The Phase-1 gate still governs interpretation:** run it FIRST, because it
supplies the λ schedule's (f_lo, f_hi) and decides whether the adaptive objective
is justified at all. The Phase-2..4 code will *run* regardless — but with
placeholder frequencies, and a NO-FLIP verdict would mean the adaptive arm has
nothing to beat.

```bash
cd ~/DASFWI && git pull && mkdir -p output logs

# ---- PHASE 1: the hypothesis gate -------------------------------------------
condor_submit hpc/condor/run.sub -a 'kind=calibrate'      # 1a, ~1-2 min/rung
cat output/run_calibrate_*.out                            # -> TRANSITION rungs
./hpc/marmousi_full_das/make_ladder_combos.sh s16 s20 s24 # use those rungs
condor_submit hpc/condor/skip_ladder.sub                  # 45 combos x rungs
python hpc/marmousi_full_das/flip_curve.py --csv flip.csv --plot flip.png
#   -> FLIP AT RUNG 'sNN'  (read f_lo/f_hi from it)   or   NO FLIP -> redesign

# ---- PHASE 2: does the adaptive objective beat both fixed arms? --------------
FL=3.0; FH=8.0        # <-- set from the flip curve
for OBJ in adaptive l2 sinkhorn; do
  condor_submit hpc/condor/run.sub -a 'kind=adaptive' \
    -a "extra=--objective $OBJ --start-rung s20 --iters 60 --flip-lo $FL --flip-hi $FH"
done
# compare results/marmousi_full_das/adaptive/*/metrics.json (ssim, mape)

# ---- PHASE 3: the transferable starting model --------------------------------
condor_submit hpc/condor/run.sub -a 'kind=starter' -a 'extra=--iters 80 --band 3.0'
cat output/run_starter_*.out     # skip@1-D vs skip@starter, vs the Phase-1 threshold

# ---- PHASE 4: the full pipeline (and its controls) ---------------------------
condor_submit hpc/condor/run.sub -a 'kind=pipeline' \
  -a "extra=--start route_b --bands 2.0,3.0,4.5,full --iters 50 --flip-lo $FL --flip-hi $FH"
condor_submit hpc/condor/run.sub -a 'kind=pipeline' \
  -a 'extra=--start linear --fixed l2 --bands 2.0,3.0,4.5,full --iters 50'   # control
```

Smoke any stage first with `--smoke` (2 iterations/band), e.g.
`-a 'extra=--smoke'`. Phase 5 (`DiagnosticLambda`, hysteresis) is implemented and
tested; enable it only if the Phase-2 logs show the frequency schedule invoking
OT when the skip diagnostic says the model is well aligned.

> The discipline that matters most: Phase 1 is a hypothesis test about strain-rate
> objectives (NIM's divergence is the standing reminder that OT-family behavior on
> this observable is not free). Nothing downstream is built until the flip curve exists.
