# DASFWI codebase map

Orientation for someone (or some model) arriving cold. **Read this, then
`ADAPTIVE_FWI_PLAN.md` for the scientific plan.**

## What this project is

Differentiable FWI of **DAS strain rate**, built on Liu Feng's ADFWI. The
mandate: recover **Vp and Vs from DAS strain rate alone**, with no auxiliary
field data, deployable at any site. Validation site is Utah FORGE (wells
78A-32, 78B-32), against Park et al., *TLE* 44(4), doi 10.1190/tle44040256.1,
who published acoustic **Vp only** — Vs is our novelty.

Two rules that are not negotiable:

- **Well logs are VALIDATION ONLY.** Never an inversion constraint, never a
  tuning target. Tuning to a log destroys the transferability claim that is the
  entire point of the project.
- **Nothing site-specific in the processing path.** Geometry, datum, bandwidth
  and stability are *measured from the data*. A hardcoded fix for FORGE is a
  second bug waiting for the next site.

## The physics that shapes the code

- **E3 gauge operator**: `ε̇ = [v_z(z+l/2) − v_z(z−l/2)] / l`, exact only when
  `l = 2·dz`. No strain→velocity conversion; the DAS layer sits between
  propagator and misfit and autograd differentiates through it.
- **Cycle skipping**: a trace is skipped when `|Δt| > T/2 = 1/(2·f_max)`. This
  is measurable from data alone (syn vs obs), which is why the adaptive switch
  transfers from synthetics to field unchanged.
- **CFL**: `dt ≤ safety·dx/vmax`, safety 0.45 — below the nominal 0.606 because
  a 17:1 air/rock contrast plus PML costs stability margin.

---

## Directory layout

```
inversion/     physics + methods, site-agnostic. THE LIBRARY.
forge/         FORGE-specific I/O, QC and figures. THE SITE ADAPTER.
hpc/standalone/  one-cell drivers + result rankers
hpc/slurm/     Bridges-2 job scripts
hpc/condor/    scheduler-agnostic dispatch (shared by SLURM and HTCondor)
tests/         pytest + the import guard
```

The split matters: **`inversion/` must never import `forge/`.** Anything in
`inversion/` should run at a new site untouched.

---

## `inversion/` — the library

| module | what it does |
|---|---|
| `config.py` | **single source of truth** for misfits, optimizers, per-misfit run settings. Change a technique here, not in a driver. |
| `adaptive_misfit.py` | `SkipSwitch` (binary λ on the measured skip fraction, with hysteresis/dwell/stall guards) and `BlendedMisfit` (gradient-norm normalised, short-circuits at λ∈{0,1}). The core method. |
| `skip_diagnostic.py` | `skip_fraction(syn, obs, dt, f_max)` — needs **no true model**, which is why the switch works on field data. |
| `das_conditioning.py` | `ConditionedMisfit`: arrival windowing + channel weighting, wrapping any misfit. Delegates unknown attributes to the inner misfit. |
| `near_surface.py` | air layer (flat or topography-following), Park's 1000–6000 bounds, anisotropic λ/4 gradient smoothing, CFL / `stable_time_axis`. |
| `fibre_geometry.py` | **measures fibre orientation from the moveout** and refuses when it disagrees with the headers. Site-agnostic. See "the geometry bug" below. |
| `device.py` | device selection that **refuses to fall back to CPU inside a GPU job** — that failure silently bills a full 8 h walltime. |
| `das_qc.py` | 4-test DAS QC: amplitude vs *shape* distortion, with a geometric premise guard that can veto the verdict. |
| `field_acceptance.py` | acceptance criteria for data with **no true model**: arrival-lag reduction, two-well cross-validation, zone alignment, log comparison. Thresholds fixed before any field result. |
| `tf_phase.py` | Fichtner time-frequency phase misfit, adjoint-verified. |
| `starting_model.py` | 1-D gradients, smoothed-truth rungs, `vs_from_vp`. |
| `metrics.py` | SSIM/MAPE etc. **Synthetics only** — meaningless on field data. |

## `forge/` — the site adapter

| module | what it does |
|---|---|
| `field_loader.py` | SEG-Y → gathers + geometry. **Reads the byte map the acquisition declares in its textual header**, not the names segyio assumes. Establishes the depth datum. |
| `preflight.py` | **15 checks on the real data in one command.** Exit 0 = safe to submit. Run this before spending any SU. |
| `traveltime_tomography.py` | first-break picking + VSP check-shot velocity + tomography. Picks the **dominant** event, not the first threshold crossing. |
| `proxy_model.py` | FORGE-geometry synthetic (declares its own inverse crime in the docstring). |
| `realistic_synthetic.py` | elastic generation → acoustic inversion, noise, wavelet mismatch. Not an inverse crime. |
| `well_logs.py` | 58-32 DSI sonic — Vp *and* **Vs (DTSM)**, which Park do not use. Validation only. |
| `plot_field_result.py` | Park-style figures. `velocity_panel()` is the **one** definition of a velocity panel: white air, blue slow → brown fast, fixed scale, depth clipped to illumination. |

## `hpc/` — running things

| script | what it does |
|---|---|
| `standalone/run_field_das.py` | **the FORGE field driver.** One cell per invocation. |
| `standalone/run_forge_synthetic.py` | FORGE-geometry synthetic with known truth. |
| `standalone/run_acoustic_das.py`, `run_elastic_das.py` | Marmousi drivers. |
| `standalone/submit_forge_field.sh` | the field campaign. `--list` / `--dry-run` / submit. Gates on preflight, tag uniqueness, and 30/150 pairing. |
| `standalone/rank_forge_field.py` | reads field results. Ranks on **data fit** (the only field-measurable quantity); validation reported separately and never sorted on. |
| `standalone/rank_forge_synthetic.py` | reads synthetic results. Ranks on depth-resolved error, never SSIM (degenerate here). |
| `marmousi_full_das/run_traveltime_starter.py` | **Route B**: wave-equation cross-correlation starting model, no picking. |
| `condor/run_standalone.sh` | maps a `kind` to a script. Both SLURM and HTCondor go through it. |
| `slurm/bridges2.sbatch` | one job → one H100 on GPU-shared (2 SU/GPU-hour). |
| `tests/check_imports.py` | every entry point must resolve its imports in a bare shell. Run with no args to sweep. |

---

## How to run the field campaign

```bash
export FORGE_DAS_DIR=/path/to/DAS_VSP
export DASFWI_OPT=nadam
PF_LOCAL=1 SWEEP=1 hpc/standalone/submit_forge_field.sh --dry-run   # gates
SWEEP=1 hpc/standalone/submit_forge_field.sh                        # submit
python hpc/standalone/rank_forge_field.py --validate                # read
```

`--dry-run` validates configuration and exits before the loop, so it **cannot**
catch runtime errors. `forge/preflight.py` is what exercises the real data.

## Deploying at a new site

1. Point `FORGE_DAS_DIR` at the new directory (wells are discovered, not named).
2. `python forge/preflight.py --data-dir <path> --dz <m> --f0 <Hz>`.
3. Fix whatever it refuses. It measures geometry, datum, CFL, amplitude,
   orientation and picks from the data itself.
4. Only `forge/plot_field_result.py` (`WELLS`, zone depths) and
   `forge/well_logs.py` carry FORGE constants, and both are figure/validation
   code — not the processing path.

---

## Failure modes this codebase has actually had

Every one of these produced **plausible numbers rather than a crash**, which is
why the preflight exists. If something looks wrong, suspect these first.

| what happened | how it presented |
|---|---|
| **Channel depths read from the wrong header field.** Byte 41–44 is `RECTVD` per the textual header; segyio calls it `ReceiverGroupElevation`. Receivers landed ~2500 m too deep *and* reversed. | 13 cells completed, loss fell, models meaningless. Skip pinned at 1.000. |
| **First-break picks on noise.** First threshold crossing took an early noise burst; the traveltime starter came out saturated at the 6000 m/s ceiling — a constant block. | `--starting traveltime` cells were *invalid tests*, not bad results. |
| **`convsi` absorbs timing error.** It is source-independent, so a systematic shift goes into the matching filter. | Loss → 1e-7, "100% reduction", model railed to both bounds. |
| **Unmasked gradient below illumination.** | Invented structure under the fibre. |
| **Denormal underflow.** DAS strain rate ~1e-16 in float32; normalisation divides by ~1e-40. | NaN. |
| **BLAS threads on a login node.** One thread per core × wall time. | Killed at 2105 of 1800 CPU-s. |
| **`PYTHONPATH` assumed.** | `ModuleNotFoundError` on the cluster only. |
| **Tag collisions** (7 times), most recently `--iterations` missing from the tag. | Two cells silently share one output directory. |

**The pattern:** each *layer* passed its own test and the *seam* between layers
went unchecked. And several diagnostics were confidently wrong before they were
right — a statistic computed over too little data (38 of 2060 traces), a
degenerate `polyfit`, an amplitude gate that kept the loudest 1.8% of a fibre.
When a check reports success, ask what it would have done on known-bad input.
