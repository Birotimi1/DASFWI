# Adaptive Frequency-Continuous DAS-FWI — Research Plan

**Status (2026-08-03).** Acoustic Route B steps 1–3 **PASSED**. Conditioning A/B
**done → negative**. TF-phase **built, adjoint-verified, campaign submitted**.
**Next: FORGE SYNTHETIC (step C), then FORGE acoustic field (step D).**
This plan was designed in dialogue (Opus) and verified mathematically (Fable).

---

## 🔁 SESSION HANDOFF — READ THIS FIRST IF YOU ARE PICKING THIS UP COLD

### Where we are

| step | state |
|---|---|
| 1. Acoustic Route B starter | ✅ `gc_adam` wins (skip 0.544→0.439). NOT traveltime. |
| 2. Acoustic NON-SKIP (i300 starter) | ✅ l2 0.846 best; switch 0.842 correctly stays out of the way |
| 3. Acoustic SKIP (i50 starter) | ✅ **switch 0.742 BEATS l2 0.616, at all four optimizers** (+0.075…+0.126) |
| A. Conditioning A/B (32 cells) | ✅ done → **NEGATIVE**, and `c` has a real flaw (below) |
| B. TF-phase | ✅ done → **LOSES to our switch**; mechanism measured (below) |
| **C. FORGE SYNTHETIC** | ⬅ **NEXT** — all code prerequisites now DONE |
| D. FORGE acoustic field | after C; runs **both** starters head-to-head |

**CODE COMPLETE 2026-08-03 (195 tests pass).** Everything the FORGE work needs
now exists:
- `inversion/near_surface.py` — air layer, Park's 1000–6000 bounds, **anisotropic**
  λ/4 smoothing (4:1 H:V). Tasks #51, #52.
- `hpc/standalone/run_field_das.py` — **the switch**, **multiscale**
  (`--bands`, `--iter-alloc`), **Route B** (`--starting route_b`), **`convsi`**
  as a refiner. None of this existed before. Tasks #46, #48.
- `forge/realistic_synthetic.py` — **elastic generation → acoustic inversion**,
  noise, wavelet mismatch. The thing that makes our four "settled negatives"
  re-testable.

> ⚠️ **FOUR NEGATIVES ARE NOW SUSPECT, NOT SETTLED.** Windowing, channel
> weighting, λ/4 smoothing and multiscale were all measured on
> `proxy_model.generate_observed` data — an **inverse crime by its own
> docstring** ("same propagator family, same operator, no noise"). They were
> tested against data containing none of the errors they exist to remove. Those
> results are valid about *synthetics* and invalid as guidance for the *field*.
> Nothing is retracted yet — but the non-crime synthetic can now falsify them,
> which it could not before. **Re-test before carrying any of them into FORGE.**

**Headline result to protect:** 0.742 from a *deployable* Route B start beats
0.664 obtained at s16 from a *smoothed-truth* start, at slightly harder skip.
The starting model we can actually build in the field outperformed the
truth-derived one.

### Why FORGE synthetic before FORGE field — do not skip it
Field FWI has **no ground truth**. A wrong geometry projection, E3 operator,
starter port or misfit choice yields a plausible model with nothing to flag it.
The synthetic proxy is the only place those fail loudly. It also validates the
Route B starter port (task #46, a prerequisite for D anyway), tests multiscale
where bandwidth exists, and picks the field misfit — `l2` is unavailable at
FORGE because the source wavelet is unknown, so it is `gc`/`convsi`/`tfphase`.

### Settled NEGATIVES — do not re-run, do not re-propose
- **L-BFGS and NLCG diverge to NaN.** Both line-search; the batched gradient is
  stochastic so Wolfe/Armijo conditions are evaluated on noise. ~54 SU burned.
  *(Consequence: the missing `ElasticFWI` closure path is now moot.)*
- **Noe conditioning does not help on synthetics.** Best cell +0.003 (noise);
  `g` costs ~0.13 everywhere. **But this is inverse-crime data** — no noise,
  perfect amplitudes — which is precisely what Noe's steps exist to fix, so it
  does **not** settle the field case.
- **`c` (channel weighting) has a REAL FLAW, and it applies at FORGE too.**
  `ConditionedMisfit` weights the *data*; the intent is to weight each channel's
  *contribution*. Equivalent only for a **quadratic** misfit. Measured: a weak
  channel (w=0.066) is scaled by 0.066 under L2 but **0.0003 under
  envelope^1.5 — 220×**. The switch's rescue *is* its envelope stage, so `c`
  gutted it: `switch+c` collapsed 0.742→0.26 while `l2+c` was unharmed at 0.614.
  **Do not enable `c` with a nonlinear misfit until it weights contributions.**
- **Multiscale "hurts" was an artefact** — see the bandwidth section below.

### TF-phase result — our switch wins, and we know *why* it wins

| | non-skip (i300) | **skip (i50)** |
|---|---|---|
| l2 | **0.846** | 0.616 |
| **switch** (envelope→l2) | 0.842 | **0.742** |
| switch **+tfphase** robust | 0.840 | **0.545–0.559** |
| tfphase standalone | 0.697–0.741 | **0.365–0.438** |

**Envelope remains the better robust stage** and the switch keeps its win.

**Mechanism, measured rather than assumed.** `dφ = −ω·Δt` wraps at
|Δt| > 1/(2f). Fichtner's weight is the envelope *amplitude*, which on a
narrowband source peaks at the **source peak** — i.e. the highest,
**first-to-wrap** rows. Fraction of total TF weight sitting on rows still
unwrapped: 30 ms → 100%, 80 ms → 100%, **120 ms → 43.6%, 150 ms → 26.2%**. At
the i50 starter `skip_fraction = 0.668`, so most traces exceed the 80 ms
full-band T/2 — **TF-phase is reading mostly folded phase**, which is worse
than useless because folded phase points the wrong way.

Two coupled causes: only **6 Gabor rows exist** (1.06 octaves), *and* amplitude
weighting starves the few unwrapped low rows. **Bandwidth is the root, the
weighting is the amplifier.**

> **Not a verdict on TF-phase in general.** Fichtner applies it to broadband
> teleseismic data where most weight would sit on unwrapped rows. This is a
> verdict on TF-phase with a **narrowband source under strong skip**.
>
> **Future work, genuinely novel:** make the TF weighting **skip-aware** —
> down-weight rows whose wrap limit falls below the *measured* lag. We already
> compute that distribution in `skip_diagnostic.trace_lags`, so it is cheap and
> connects directly to the switch.

### The one fact that explains two "failures"
**Marmousi is 1.06 octaves and GRID-CAPPED; FORGE field is 4.4.** Both
multiscale and TF-phase depend on low-frequency content that our synthetic
barely has. Do **not** propose "raise F0 on Marmousi": a 40 m grid resolves
~3.8 Hz at 10 ppw and f90 is already 6.25 Hz, so it needs dx≈2.5 m — ~256× cost
and regenerating the observed data invalidates the whole board.

### ✅ THE FIELD RECIPE, DECIDED BY MEASUREMENT (2026-08-04)

FORGE synthetic: elastic data, acoustic inversion, 162 m ramp, noise, wrong
wavelet. Solo arms, so the misfit under test actually runs.

> ## `convsi` + WINDOWING + STOP EARLY

| refiner | shallow | moved% | dFit | **fit / % moved** |
|---|---|---|---|---|
| **convsi + win** | 209 | **3.88** | **+78.8%** | **20.3** |
| convsi | 263 | 6.16 | +66.3% | 10.8 |
| l2 + win | **196** | 4.72 | +31.6% | 6.7 |
| l2 | 197 | 4.91 | +23.4% | 4.8 |
| gc + win | 235 | 6.38 | *(3.09 → −315)* | — |
| gc | 261 | 7.33 | *(5.52 → −345)* | — |

**`l2` has the lowest shallow error and that is NOT a win.** It moves the model
*more* than `convsi+win` while fitting the data **2.5× worse** — its model is
barely constrained by the data. The deciding metric is **fit per unit of
movement**, where `convsi+win` is **3× better**. Exactly what theory predicts:
`convsi` is source-independent so a wrong wavelet cancels, whereas `l2` fits
amplitude *and* phase and cannot reconcile one.

**Park's `gc` is the worst of the three** under a wrong wavelet (235–261). They
assume a 10 Hz Ricker and never test the sensitivity.

**Windowing helps all three, paired**: convsi −54, gc −26, l2 −1 m/s.
**Prediction confirmed** — recorded in writing before any FORGE run.

> ### ⏱ AND THE BIGGEST PRACTICAL FINDING: EARLY STOPPING BEATS EVERY MISFIT
> Shallow error grows **monotonically** with iterations while the loss falls —
> the acoustic code inventing near-surface velocity to explain **surface waves
> it cannot model**. 30-iteration cells reach shallow **173–186**; every
> 150-iteration cell is ≥ 196. **Park's 30 iterations may be protective, not a
> budget limit.** Nothing eliminates the damage; windowing and early stopping
> only reduce it.

**METRIC WARNINGS.** SSIM is **degenerate** here — it falls monotonically from
iteration 0, so ranking on it says "never invert"; rank on depth-resolved error.
And no percentage change for `gc`, whose correlation loss crosses zero.

### 🌊 SURFACE WAVES AT FORGE — and a prediction that reverses a negative result

Park: *"Due to the strong **surface waves** present in the **near-offset** data
(S2 to S4), the reflected waves are marked only in the far-offset data."*

Rayleigh waves are **elastic**; an **acoustic** FWI cannot model them, so they
get fitted by **spurious velocity structure** unless muted. The tool for that is
our **arrival windowing** (`w`, `das_conditioning.arrival_window`) — which
*hurt* on clean synthetics (−0.028 at skip) for the obvious reason that there
was no noise or coda to remove.

> **FALSIFIABLE PREDICTION, recorded before the run: `w` HELPS at FORGE where it
> HURT on Marmousi.** If it doesn't, the windowing implementation is wrong, not
> the idea. This is the clearest case yet of why a synthetic negative does not
> transfer: the conditioning A/B tested noise-removal tools on noiseless data.

### 📉 Three concrete gaps from their INV1/INV2 parameters

| | Park | us | consequence |
|---|---|---|---|
| **Vp lower bound** | INV2 **1.0** km/s | **1.5** km/s | alluvium can be ~1 km/s, so our clamp **forbids the true near-surface** and the error is absorbed elsewhere (task #52) |
| **Gradient smoothing** | **200×100 m**, **100×25 m** (anisotropic, 2:1 and 4:1 H:V) | one **isotropic** span | we smooth vertically as hard as horizontally, discarding the **vertical resolution that is the DAS-VSP's whole advantage** (task #52) |
| **Air layer** | yes | **no** | see task #51 |

All three are near-surface faults, and they compound. The `g` conditioning cost
~0.13 SSIM on Marmousi — an isotropic span may be part of why, so test
anisotropic before writing wavelength-scaled smoothing off.

### ✅ A truth-free acceptance number we can match exactly

INV2 reduced the **first-arrival time mismatch by 51.7%**. That needs no true
model, and it scores **our Route B starter on the same axis as their manually
picked tomography** — the cleanest possible head-to-head for the transferability
claim (task #49).

### 📐 PARK'S FULL WORKFLOW — what it confirms, and one gap on our side

**VM0 = Miller (2019)** → **INV1** traveltime tomography on *3D surface
seismic* → **INV2** traveltime tomography on *2D DAS-VSP* → **VM2** →
**INV3 = acoustic FWI** → **VM3**, validated against well logs.

**1. They MANUALLY PICK first arrivals** — on both datasets. That is exactly
what Route B (wave-equation xcorr, no picking) avoids, and it is the core of our
transferability claim. Their eikonal stack — our documented fallback — is Noble
et al. 2014 (hybrid plane-wave/spherical), Zhao 2004 (fast sweeping), Tong 2021
(adjoint-state traveltime tomography, no ray tracing).

**2. FWI misfit = global correlation norm** (Choi & Alkhalifah 2012), for
"phase information… reducing sensitivity to amplitude mismatches"; gradients by
adjoint-state (Plessix 2006). **No multiscale is mentioned in the methodology
*or* the parameter passage** — two independent passages, both silent. Read as
single-band, but confirm against the paper before publishing the claim.

**3. ⚠️ TOPOGRAPHY — A GAP ON OUR SIDE (task #51).** Park: *"The inclined nature
of the surface topography requires careful handling… we set the highest point of
the model as depth 0 and incorporate an **AIR LAYER**."* We match the datum
convention (`field_loader`: datum = max source elevation) **but have no air
layer** — `VP_BOUND=(1500,6000)` and a flat `free_surface=True` make everything
between the datum and the true ground surface **rock at ≥1500 m/s**. Every
source below the highest then radiates through fictitious rock, and the
inversion will absorb the error by **lowering near-surface velocities** —
corrupting the shallow zone the DAS-VSP exists to constrain. **Measure the
relief first**; if it is small relative to a wavelength this may be negligible.

**4. How much prior information their VM2 carries:** a published regional model
+ 3D surface seismic + manual picks on two datasets. **Our Route B starter uses
the DAS-VSP alone.** If we reach comparable quality from far less input, *that*
is the result — and a fairer framing of our contribution than "our misfit is
better".

### 🎯 PARK'S EXACT FWI SETUP — match it, then beat it fairly

Their INV3: **10 m grid, dt 1 ms, 2 s record (nt = 2000), initial model VM2,
BOTH boreholes 78A-32 + 78B-32 combined, Ricker 10 Hz peak / 20 Hz max,
FIXED step length 0.1 km/s, 30 iterations.**

| what it settles | consequence for us |
|---|---|
| **Grid = 10 m** | confirms the choice I'd recommended on cost. Our cost there: **12.7× a Marmousi cell** → ~8 SU/cell at 300 iters, **~0.8 SU/cell at their 30** |
| **Source is a 10 Hz Ricker** | it is *assumed*, not measured — so match it for comparability, but `convsi` (source-independent) is a genuine **improvement** over their setup, not a workaround |
| **Both wells combined** | the Park-comparable run uses both. Our 78A-vs-78B cross-validation becomes an *extra* check, not the primary one |
| **Fixed step, 30 iterations** | steepest descent, no line search, no Adam |

> ⚠️ **THE COMPARISON TRAP.** Their optimizer is weak by our standards and they
> run 30 iterations to our 300. **Beating VM3 therefore does NOT by itself
> validate the switch** — we'd be beating their *budget and optimizer*, not
> their method. To claim the METHOD won, match iterations and optimizer, or
> report both matched and unmatched runs. Otherwise the headline result is
> unfalsifiable and a reviewer will say so.

**BANDWIDTH CORRECTION.** My "FORGE = 4.4 octaves" was the *raw* spectrum
(6–130 Hz) and is **not** usable for FWI: a 10 m grid caps near 20–30 Hz and
Park stops at 20. The real ladder is **3–20 Hz = 2.74 octaves**. Still a
genuine cascade — 2.6× Marmousi's 1.06 — but quote 2.74, not 4.4.

### ⚠️ FORGE READINESS — TWO BLOCKERS FOUND 2026-08-03

**`run_field_das.py` supports NONE of our method.** Zero occurrences of
`SkipSwitch`, `BlendedMisfit`, `cutoff_freq`, `--arm`, `--bands` — it runs a
single fixed misfit end to end. So at FORGE today we have neither the adaptive
switch nor multiscale. **Task #48.** Good news: `skip_fraction` needs only
syn vs obs, no true model, so the switch *does* work without ground truth.

**And multiscale IS available at FORGE** — 4.4 octaves makes a 5/8/12/20 Hz
ladder a genuine cascade. The Marmousi "cascade hurts" result does **not**
transfer (see the retraction below). FORGE is the first dataset where the
cascade and the switch might be complementary rather than redundant, since the
switch measures skip *per band*.

**Acoustic FORGE recipe — and an inconsistency I had to correct.** I wrote
"`l2` is OUT at FORGE". **That was too strong**: Park *did* invert with an
**assumed** 10 Hz Ricker and it produced VM3. The accurate statement is that L2
fits amplitude *and* phase, so a wrong wavelet is **absorbed into the velocity
model** with nothing flagging it — a vulnerability, not an impossibility.

> ⛔ **THE TRANSFER GAP THIS EXPOSES (task #50).** Our headline result uses the
> **L2 refiner** (`envelope → l2`, 0.742). The obvious substitute **loses on
> Marmousi** (`switch-gc` 0.585), and the principled one — **`convsi`,
> source-independent — has never been tested as a refiner and is not even in
> `--refiner` choices** (`l2`, `gc`, `tfphase`). So "the switch wins" and "L2 is
> questionable at FORGE" cannot both stand unexamined: the *specific winning
> configuration* rests on the misfit whose FORGE validity is in doubt.
>
> **Decide it by experiment, on the FORGE synthetic, where we control the
> wavelet AND have truth:** generate with a true wavelet, invert with a
> deliberately wrong one (Park's 10 Hz Ricker), compare `l2`/`gc`/`convsi`
> refiners ± the switch, and quantify the velocity error. Include a
> correct-wavelet control to separate wavelet error from everything else.
> Publishable on its own: Park assumed a Ricker and never tested the sensitivity.
> *Watch out:* `convsi` runs with `batch_size=2, checkpoint_segments=2,
> normalize=False` — unlike `l2`/`gc` — so the switch's gradient-norm
> normalisation and run settings must be **checked, not assumed**.

Conditioning: `w` and `g` are defensible on real data; **`c` stays OFF** until
it weights contributions rather than data.

**Two real risks, both bigger than the misfit choice:**
1. **The source wavelet is UNKNOWN** — currently a placeholder Ricker at
   F0=15 Hz. `convsi` sidesteps it; anything else silently inherits a wrong
   wavelet.
2. **Acceptance must be defined in advance (task #49) — and FORGE is NOT
   truth-free, which I got wrong.** Park et al. validated VM0 (initial) vs VM3
   (final) against **WELL LOGS**:
   - **Borehole 58-32** — a *third* well, distinct from the DAS wells — carries
     a **SONIC LOG (direct Vp)** plus **drill-cuttings density** ('B' bulk,
     'M' matrix). Density defines zones **I / II / III = unconsolidated
     alluvium / consolidated alluvium / GRANITOID BASEMENT**.
   - **78B-32** carries a simplified **lithology log** marking the granitoid
     transition.

   So the primary test is: does recovered Vp reproduce the 58-32 sonic log, and
   do the velocity boundaries line up with the I/II/III zone boundaries and the
   granitoid transition? Secondary: data-residual reduction, our own
   skip-fraction falling, and **cross-validating 78A-32 against 78B-32**
   (independent data, shared geology → models should agree where they overlap).

   > ⛔ **METHODOLOGICAL LINE.** Our mandate is Vp **and** Vs from DAS strain
   > rate **alone**. Logs are **VALIDATION ONLY — never an inversion constraint
   > and never a tuning target.** Sliding from "validate against the log" to
   > "tune until it matches" would destroy the transferability claim, which is
   > the entire point of the method. **Fix the acceptance thresholds before
   > looking at the comparison.**

   *First confirm we actually hold the 58-32 logs — the network share was
   unreadable when checked.*

### Recurring failure modes — check these before every launch
1. **Tag collisions — five occurrences.** Every knob that changes an experiment
   must be in the output tag. Verify tags are distinct *and* don't collide with
   the existing board before submitting.
2. **`--dry-run` cannot catch runtime errors.** It exits before the loop. The
   16-cell conditioning wipeout was two runtime bugs. **Always smoke.**
3. **Smoke must cover every distinct code path.** Every non-ladder arm builds a
   `BlendedMisfit`, so `l2` drives `set_lambda` too — smoking only `switch`
   left half the campaign unverified.
4. **A job that exits 0 is not a job that worked.** Check `metrics.json` exists
   *and* its contents (`losses_finite`, `diverged`, the recorded terms).
5. **zsh does not word-split unquoted `$var`** — has produced false failures in
   my own test harnesses three times. Use `${=var}` or explicit invocations.
6. **The scratch repo gets purged** — see the restore section at the bottom.

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
cells, s6/s16/s20, 5 optimizers — switch wins everywhere); the elastic
regression (0.583/0.702, reproduces the campaign — code validated).

> ⚠️ **RETRACTED 2026-08-03:** this list used to say *"acoustic multiscale
> (NEGATIVE, the cascade hurts)"*. **That was an artefact of the setup, not a
> property of frequency continuation**, and it was repeated as settled for over
> a week. Two causes, both fixed: the default ladder `3.0,4.5,6.25,full`
> clamped two bands to the same 6.25 Hz (a quarter of the budget inverting
> identical data — the code printed a NOTE nobody read; preflight now refuses
> it), and `--iters` was per band with an **equal** split, so the cascade got
> 75 iterations at the band that sets the score while the single-scale control
> got 300 (`--iter-alloc final-heavy` fixes this). Underneath both: **Marmousi
> is 1.06 octaves**, so it cannot test a cascade at all. The real test is the
> FORGE synthetic / field, at 4.4 octaves.

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
│   ├── tf_phase.py           Fichtner TF-PHASE misfit (Gabor plane, phase only).
│   │                         Registered as "tfphase"; ADJOINT-VERIFIED against
│   │                         finite differences before it was allowed to score.
│   ├── das_qc.py             SITE-AGNOSTIC FIELD QC -- amplitude vs SHAPE
│   │                         distortion. qc_das(gathers, dt) is the one call;
│   │                         auto_band, spacing_is_adequate, recommended_settings,
│   │                         format_report. RUN AT EVERY NEW SITE (see below).
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

## 🔎 STEP 0 AT EVERY FIELD SITE — DAS waveform-shape QC (mandatory)

Before inverting **any** field DAS dataset, run `inversion/das_qc.py`. It answers
the one question that changes the whole strategy, and it is the reason this
method is transferable rather than FORGE-specific.

| answer | meaning | what to do |
|---|---|---|
| **AMPLITUDE-ONLY** | coupling acts as a **scalar** | nothing new — per-trace normalisation (already on), channel weighting, `gc`/phase misfits are sound |
| **SHAPE DISTORTION** (repeatable) | coupling acts as a **per-channel FILTER** | **no misfit choice fixes this.** Estimate the transfer function from the spectral ratios and deconvolve, or model coupling in the forward problem (Celli et al. 2024) |
| shape mismatch, not repeatable | noise, not the fibre | channel weighting + arrival windowing suffice |
| **INCONCLUSIVE** | the neighbour premise fails here | do **not** conclude distortion — retry with `n_neigh=1`, a lower band, or more shots |

**Why it works anywhere:** channels are metres apart, wavelengths are ~100 m, so
neighbours *must* agree. It needs only the observed gathers — no true model, no
source wavelet, no second instrument. The decisive statistic is whether the
ratio to neighbours is **flat** (a scalar) or **structured** (a filter), tested
in *both* magnitude and phase:

| # | test | catches |
|---|---|---|
| 1 | residual after xcorr alignment + normalisation | any shape difference |
| 2 | spectral ratio `\|D_i(f)\|/median_j\|D_j(f)\|` — flat or tilted? | **magnitude** filtering |
| 3 | repeatability across shots | separates the *fibre* from *noise* |
| 4 | cross-spectrum **phase** — linear in f, or curved? | **dispersion** |

**Test 4 is not redundant.** Test 2 sees only `|R(f)|`, so an **all-pass** filter
— flat magnitude, curved phase — passes it while genuinely deforming the
waveform, and dispersion is the one distortion a *phase* misfit cannot absorb
either. Verified: on a synthetic all-pass case test 2 reports 0% tilt while
test 4 flags it.

> ⚠️ **Test 4's statistic must be SIGNED (curvature), never an RMS departure.**
> The first version used RMS, which is unsigned and **saturates** — noise-driven
> phase scatter then reads as "consistently ~0.5 rad in every shot" and passes a
> std-below-magnitude repeatability test. It flipped **both FORGE wells to a
> false SHAPE-DISTORTION verdict** (0.4975 / 0.5421 rad, ~50% of channels — the
> median landing exactly on the threshold was the tell). Signing the statistic
> drops FORGE to **0.0040 / 0.0046 rad**, a 125× fall, because random curvature
> cancels in the median. Repeatability is likewise judged against the median's
> **standard error**, so a handful of shots cannot fake consistency.
> Pinned by `test_phase_noise_is_not_mistaken_for_dispersion`.

**The deployment guard:** that premise can fail at a site with coarse channel
spacing, strong scattering, or poor SNR — and a failed premise looks exactly
like shape distortion. So it is *tested*, not assumed: the comparison is
repeated in the bottom third of the band, where neighbour coherence is most
strongly guaranteed. Disagreement there ⇒ **INCONCLUSIVE**, never a false alarm.

Its threshold is **calibrated, not guessed** — low-band neighbour correlation
measures 1.00 clean, 0.99 scalar, 0.99 magnitude-filtered, **0.69 severe
dispersion**, **0.18 genuinely incoherent**. It must sit *below* what real
distortion leaves and *above* incoherence, hence **0.40**. An initial 0.80
silently downgraded true dispersion to "inconclusive", because severe distortion
depresses this correlation too; `test_premise_threshold_separates_*` pins it.
Note the *safe* failure direction: an extreme distortion that drove correlation
under 0.40 would read "inconclusive — go look", never "clean".

A second trap, also pinned: for incoherent traces the cross-spectrum phase is
random, so its non-linearity **saturates in every shot** — large *and* stable,
which mimics a repeatable filter. Test 4 therefore requires neighbour coherence
(`corr > 0.3`) before it will call dispersion: you can only claim a waveform is
deformed relative to its neighbours if there *is* a shared waveform. That gate
sits between measured values (0.15 incoherent, 0.45 mildly dispersive) and
**cannot be raised much** — at 0.5 the detection window closes completely,
because by the time curvature clears 0.5 rad the correlation has already fallen
below the gate.

**Two limits worth knowing when reading the output**, both covered by tests
rather than left implicit:
- **Severe** dispersion destroys neighbour correlation outright, so it is
  reported by test 1 as a shape mismatch (or by the guard as inconclusive)
  rather than by test 4. It is never silently called clean.
- A defect spanning a **contiguous run** of channels wider than the neighbour
  window is invisible by construction — the reference is distorted too. Widen
  `n_neigh`, or compare against a different fibre section.

```bash
# any site:  [S, nt, C] gathers in an .npz (band is measured if not given)
python inversion/das_qc.py --npz site.npz --channel-spacing 4.0 --v-min 1800
# FORGE:
FORGE_DAS_DIR=/path/to/DAS_VSP python inversion/das_qc.py --well 78A-32 --shots 20
```
Also runs automatically inside `hpc/standalone/run_field_das.py`
(`--qc on|off|strict`; `strict` **aborts** on shape distortion) and writes
`das_qc.json` next to the results. Exit code 2 = shape distortion.
Validated by `tests/test_das_qc.py` against six constructed cases whose answer is
known analytically (clean / scalar / fixed magnitude filter / **all-pass
dispersion** / shot-varying filter / incoherent channels), each checked with the
premise guard both on and off.

**MEASURED AT FORGE (2026-08-02, 20 shots/well): AMPLITUDE-ONLY, both wells —
now on all four tests.**

| | 78A-32 | 78B-32 |
|---|---|---|
| channels / shots | 1010 / 20 | 1206 / 20 |
| neighbour correlation (median) | 0.954 | 0.954 |
| spectral-ratio slope \|·\| | 0.023 | 0.018 |
| **frac_spectral_tilt** (magnitude filtering) | **0.0%** | **0.0%** |
| phase curvature, signed (median \|·\|) | 0.0040 rad | 0.0046 rad |
| **frac_phase_nonlinear** (dispersion) | **0.4%** | **0.33%** |
| shape mismatch | 4.4% | 1.2% |
| premise low-band corr | 0.978 HOLDS | 0.979 HOLDS |
| dead channels | 0 | 0 |

Not one channel of **2216** shows a frequency-dependent *magnitude* ratio to its
neighbours, and the *phase* curvature sits at 0.004 rad — indistinguishable from
a clean synthetic (0.0016 rad) and ~125× below the 0.5 rad threshold. **Coupling
at FORGE acts as a SCALAR, not a filter, in either magnitude or phase.**
Consequence: phase-based misfits (`gc`, TF-phase) are sound here rather than a
compromise; per-trace normalisation + channel weighting + `gc` suffice; no
deconvolution and no Celli-style coupling modelling required. Neither published
FORGE paper characterised this — Park never mentions coupling, Noe names it as a
limitation without measuring it.

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

---

## 🛟 RESTORING THE WORKING TREE AFTER IT GETS DELETED

**This has happened three times.** The local copy lives under `/private/tmp/`,
which **macOS purges by age**, so tracked files and loose git objects vanish
while `HEAD` stays intact. The first two occurrences were misdiagnosed as "git
corruption". On 2026-08-03 it took **180 tracked files** and 5 git objects.
Tell-tale sign: the files that *survive* are the ones edited that same day.

> ### ⛔ THE DANGER
> `git add -A && git commit` in that state **commits 180 deletions**, and
> pushing **wipes the cluster repo on its next pull**. This has bitten the
> project before.
>
> **RULE: before any `git add -A`, run `git ls-files -d | wc -l`.
> If it is non-zero, STOP and restore — do not stage.**

```bash
cd <repo>
# 1. diagnose
git status --porcelain | awk '{print substr($0,1,2)}' | sort | uniq -c
git ls-files -d | wc -l                          # deleted — want these back
git diff --name-only --diff-filter=M | wc -l     # MUST be 0 for a lossless restore

# 2. restore ONLY the deleted paths. By construction this cannot clobber a
#    modified file — unlike `git restore .`, which is tree-wide and is (rightly)
#    blocked by Claude Code's auto-mode classifier as irreversible destruction.
git ls-files -d -z | xargs -0 git restore --

# 3. if it errors "unable to read sha1 file", the OBJECT STORE is damaged too:
git fetch --refetch origin                       # redownload all objects
git ls-files -d -z | xargs -0 git restore --     # then retry

# 4. VERIFY — do not assume
[ -z "$(git diff --name-only origin/main)" ] && echo "IDENTICAL to origin/main"
git ls-files | while IFS= read -r f; do [ -e "$f" ] || echo "MISSING $f"; done
python -m pytest tests/ -q | tail -2             # functional proof
```

Leftover `git fsck` complaints about the reflog or a pack index afterwards are
**cosmetic** — content correctness is proven by the `origin/main` diff being
empty. Not worth re-cloning.

**Why nothing is ever actually lost:** every change is committed and pushed to
`origin/main` immediately. **The scratch copy is disposable**; GitHub and the
Bridges-2 checkout (`/ocean/projects/ees260010p/brotimi/DASFWI`) are
authoritative, and the cluster clone is unaffected by this. That push habit is
what makes this a two-minute recovery instead of a lost day.

---

# STATUS 2026-08-07 — FORGE FIELD: root causes found, one clean run pending

See `CODEBASE.md` for the script-by-script map.

## What the first field campaign actually showed

13 cells completed with falling losses and produced **nothing usable**. Three
root causes, all found by inspecting figures rather than metrics:

1. **Channel depths were wrong.** Byte 41–44 is `RECTVD` per the acquisition's
   own textual header; segyio names it `ReceiverGroupElevation`. Receivers were
   placed at 2492–3522 m and reversed. True: 0–1013 m (78A-32), 0–1209 m
   (78B-32) — confirmed independently by moveout physics AND by Park's text
   ("1010 and 1206 recording channels spaced 1 m apart to 1 and 1.2 km depth").

2. **First-break picks were on noise.** They sat at 0.05–0.25 s while the
   arrival was at 0.42–0.65 s, so the `traveltime` starter came out saturated
   at the 6000 m/s ceiling — a constant block with no information. **Groups B
   and C were invalid tests, not bad results.**

3. **The gradient was unmasked below illumination**, free to invent structure
   under the fibre.

## >>> THE RESULT THAT MATTERS: ROUTE B WORKS ON FIELD DATA <<<

With the geometry fixed, the **Route B wave-equation cross-correlation starter
— which uses NO picked first breaks —** produces a physically sensible model:
~1200 m/s at surface → 3000 at 0.4 km → 4000 at 0.75 km → 5500 below 1 km.
**That is Park's three-zone structure**, from a starter that needs none of
their ~100 hand-picked shot gathers.

The pick-based tomography starter, by contrast, was a saturated block. On this
data Route B did not merely match picking — picking failed outright.

**Open problem:** the *inversion* then degrades that good starter, adding
vertical fingering at 0.25–0.75 km and blowing up where illumination dies. The
starting model currently resembles Park more than the inverted model does.

## Changes made in response (all free, none yet run)

- λ/4 **gradient smoothing enabled** on every cell, with an unsmoothed control
  (group G) so the effect is measured. Was implemented and never switched on.
- **Gradient masked below the deepest receiver**, tapered over λ/2.
- **Plotting matches Park**: white air, blue slow → brown fast, fixed scale,
  depth clipped to illumination. One shared `velocity_panel()`.
- **Picker fixed**: dominant event + walk back to onset. Scatter 75.5 → 0.62 ms.
- **Preflight now 15 checks**, including fibre orientation and pick coherence.

## Next: ONE campaign, 18 cells, ~63 SU

Groups A/B (Route B vs tomography — the transferability claim), C (Park
baseline), D (78B-32 cross-validation), E (ablations), F (switch), G
(unsmoothed control). Every config at **both 30 and 150 iterations**, enforced
by the submit gate.

If smoothing does not fix the artifacts, Tikhonov/TV is wired in ADFWI
(`regularization_fn` on `AcousticFWI`) and unused — that is the next lever, not
another optimizer sweep.

## Known open questions

- **DAS QC is INCONCLUSIVE**, not clean: the neighbour span is 2.57 wavelengths
  at 128 Hz because channels are grid-decimated. Must be run on the **native
  1 m fibre** over the inversion band before any coupling claim.
- **Park invert both wells together** (~2216 channels/shot); we invert one well
  at ~103. Their illumination is far better and this may be why their sections
  are smoother.
- **Multiscale untested on FORGE.** The Marmousi negative may not transfer:
  Marmousi is 1.06 octaves and grid-capped, FORGE is 4.4.
