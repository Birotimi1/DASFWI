# How to run DASFWI locally (without Claude)

Everything below runs on this Mac from the repo root:

```bash
cd /Volumes/AS-Filer/EES/jbrussel/SharedData/DAS/DATA/2022_Stimulation/Codes/DASFWI
conda activate dasfwi          # or use the full path shown below
```

The conda env `dasfwi` (exported in `env.yml`) already has a `.pth` file so
`import ADFWI` and `import das/forge/inversion` work from anywhere. If you
ever rebuild the env: `conda env create -f env.yml`, then re-create
`<env>/lib/python3.10/site-packages/dasfwi_paths.pth` containing two lines —
the ADFWI repo root (`.../Codes/ADFWI`) and this repo (`.../Codes/DASFWI`).

If `conda activate` is unavailable in a script, call the interpreter
directly: `/Users/birotimi/anaconda3/envs/dasfwi/bin/python`.

## 1. Test suite (always run after changing code)

```bash
python -m pytest tests -q                 # full suite, ~6 min (59 tests)
python -m pytest tests -q --deselect tests/test_inverse_crime.py::test_miniature_inversion
                                          # fast subset, ~45 s
```

The suite contains the project's correctness gates (E6–E9 in
`tests/test_adjoint.py`, incl. the AD-vs-FD gradient check) and the
bit-identical no-regression test for the ADFWI patch. All in float64/CPU.

## 2. Acoustic Marmousi2 demo (single run)

```bash
python inversion/run_marmousi_demo.py     # ~45 min CPU
```

Downscaled 1.5 x 1.0 km Marmousi2 crop (x 10–11.5 km), 5 m grid, vertical
DAS fiber at x = 750 m, 6 surface shots, multiscale 5 → 10 Hz, 25 iterations.
Results (npz + png) → `results/marmousi_demo/`.

Knobs (edit the constants / config dict in the script):
- crop position `X0_M, Z0_M`; grid `NZ, NX`
- smoothing of the starting model: `gaussian_filter(vp_true, sigma=...)`
  (sigma in grid cells; 5 m per cell)
- `optimizer=` `"sgd"` (default here; gradient-proportional classic FWI
  update, peak step lr·vmax m/s per iteration) or `"adamw"` (moves every
  cell ~lr m/s per iteration — beware noise amplification in regions the
  fiber does not illuminate; see commit 1ab7cd2)
- `bands=[(cutoff_Hz, iterations), ...]` multiscale ladder
- `rho=` fixed density shared by BOTH observed and inversion models
  (never derive obs-rho and inversion-rho from different vp's)

## 3. Acoustic optimizer x misfit matrix

```bash
python inversion/run_acoustic_matrix.py --quick   # ~15 min debug pass
python inversion/run_acoustic_matrix.py           # full, ~4–6 h CPU
```

Runs {adam, sgd} x {gc, weci, sinkhorn} on the identical Marmousi setup:
- `gc` — global correlation (dtype-safe `GCMisfit64`)
- `weci` — hybrid Envelope→GC sigmoid schedule (our T6 `ScheduledMisfit`
  with the code-verified WECI weight `1/(1+exp(-(i-N/2)))`, N = total iters)
- `sinkhorn` — Wasserstein sinkhorn divergence via `SinkhornSafe`, using
  EXACTLY Liu's misfit parameters from the ADFWI examples
  (`dt=0.01, sparse_sampling=2, p=1, blur=1e-2`; see
  `examples/acoustic/02-misfit-functions-test/01-Marmousi2-Test/`).
  `SinkhornSafe` additionally fixes/handles three things Liu's setup never
  encounters (surface receivers always record signal; CUDA casts dtypes):
  1. upstream dtype mixing that crashes on CPU in every precision;
  2. scaling: it is run with `waveform_normalize=False` and applies ONE
     detached global scale per shot instead — per-trace max-normalization
     divides numerically-dead fiber traces (~1e-35 precursor noise) by
     ~1e-38 maxima and its BACKWARD overflows float32 into a full-grid
     NaN gradient;
  3. a relative dead-trace mask (>= 1e-3 of the gather peak on BOTH obs
     and syn) so no-information traces never enter the transport problem.
  Do NOT swap in guessed blur/p/dt values — the p=2/blur=0.1/dt=true-dt
  combination NaN's geomloss's epsilon schedule on these gathers.

Liu's example practices adopted for all combos: `torch.optim.Adam` (the
"adam" optimizer option), `waveform_normalize=True`, and a gradient mask
zeroing the top 10 rows (`grad_mask_top=10`, suppresses source artifacts).

Per-combo learning rates are in `OPTIMIZERS` at the top of the script
(remember the different lr semantics per optimizer: Liu runs Adam lr=10
over 300 iterations with dense surface receivers; the matrix uses 2.0 for
25 iterations of single-fiber acquisition). Results →
`results/acoustic_matrix/<combo>/result.npz`, plus `summary.json`
(RMS init/final, update correlation — total and near/far the fiber — and
runtimes) and the comparison figures `matrix_vp.png` / `matrix_losses.png`.

## 4. Elastic Vp+Vs Marmousi2 demo

```bash
python inversion/run_marmousi_elastic_demo.py     # ~9 h on Apple GPU (MPS)
```

Joint Vp+Vs inversion from vertical-fiber DAS strain rate; 180 m Gaussian
initial models; 300 iterations (150 @ 5 Hz + 150 @ 10 Hz); rho fixed at
truth. Auto-selects the Apple GPU (`mps`, validated to match CPU gradients)
and falls back to CPU (~4x slower). Checkpoints every 10 iterations →
`results/marmousi_elastic_demo/checkpoint.npz`; final results + figures in
the same folder.

Two hard-won constraints baked in — do not remove:
- `MIN_VP_VS = 1.5`: after each update, `vs <= vp/1.5` is enforced.
  Without it, cells drift below vp/vs = sqrt(2) (negative Poisson's ratio)
  and the elastic scheme destabilizes (first 300-iter run diverged this way).
- `GCSafe` clamps trace-norm divisions (near-zero traces underflow float32
  into NaN).

## 4b. Full-Marmousi2 DAS campaign (HPC; scripts in `hpc/marmousi_full_das/`)

The 30-run campaign (6 misfits x 5 optimizers, Liu's exact Marmousi2 setup
with receivers replaced by 4 vertical DAS fibers, 80 m gauge on the 40 m
grid) is designed for HPC. Local machines only VERIFY the wiring:

```bash
python hpc/marmousi_full_das/generate_obs.py          # once (~1 min CPU)
python hpc/marmousi_full_das/run_one.py --misfit gc --optimizer adam --smoke
python hpc/marmousi_full_das/run_one.py --misfit sinkhorn --optimizer sgd  # full run: HPC only
```

- combos manifest: `hpc/marmousi_full_das/combos.txt` (one "misfit
  optimizer" per line, for array jobs)
- shared setup (Liu-verbatim constants, fibers, paths): `common.py`;
  paths override via `ADFWI_ROOT`, `MARMOUSI_DIR`, `DASFWI_RESULTS`
- Marmousi2 SEGY data lives OUTSIDE the repo (`../Data_downloads/marmousi2`,
  never pushed); the repo's `ADFWI_local/` makes it self-contained on HPC
- submission steps live in the LOCAL-ONLY `HPC_SUBMISSION.md` (gitignored)
- optional per-run regularization: `--regularization
  tikhonov1|tikhonov2|tv1|tv2` (Liu's 04-example settings)
- device support (auto-picked cuda -> mps -> cpu; `--device` overrides):

  | misfit | CUDA (HPC) | MPS (this Mac) | CPU |
  |---|---|---|---|
  | l2, gc, sinkhorn | yes | yes | yes |
  | envelope, weci | yes | NO (MPS lacks complex FFT for the Hilbert transform) | yes |
  | sdtw | yes | NO (pysdtw is cuda/cpu only) | yes (via `SdtwSafe`; upstream `Misfit_sdtw` demands CUDA on any build — device-vs-string comparison bug) |

  So local smokes for envelope/weci/sdtw need `--device cpu`; all six run
  natively on the HPC's CUDA GPUs (Liu's own environment).

## 4c. Standalone one-file scripts (novice-friendly; acoustic and elastic)

For anyone who wants to run ONE inversion without touching the campaign
machinery — Liu-example style, every parameter in a marked block at the top
of the file, linear flow, one file to read:

```bash
python hpc/standalone/run_acoustic_das.py --misfit gc --optimizer adam
python hpc/standalone/run_acoustic_das.py --conventional   # pressure-receiver
                                                           # control (A/B)
python hpc/standalone/run_elastic_das.py  --misfit gc --optimizer adam
# add --smoke to either for a 2-iteration wiring check
```

- Acoustic: Liu's Marmousi2 acoustic setup, DAS strain rate on 4 vertical
  fibers (80 m gauge), patched AcousticFWI; `--conventional` flips to Liu's
  original surface pressure line — the direct observable A/B on identical
  everything else.
- Elastic: Liu's iso-elastic Marmousi2 setup (45 m grid, f0=3 Hz, constant
  rho=2450, water rows pinned+masked, fd_order=4); the inversion loop is in
  the file (ElasticFWI is read-only upstream) and includes the mandatory
  Poisson clamp (vs <= vp/1.5).
- Both: same six misfits and five optimizers as the campaign, Liu's exact
  constructor settings; model source switchable in one marked block
  (Marmousi2 / synthetic / field data — see CASE A/B/C comments).
- Portable misfit fixes now live in `inversion/safe_misfits.py`
  (GCMisfit64, SinkhornSafe, SdtwSafe) and are shared by all runners.

## 4d. GPU strategy (following Liu's ADFWI practice)

ADFWI is SINGLE-GPU per process: Liu runs his test matrices by pinning one
experiment per card (`device="cuda:0"..."cuda:7"` across his examples) and
launching them concurrently; the propagators' `gpu_num`/`cpu_num` arguments
are stored but never used (multi-GPU is upstream future work). Ours follows
the same one-process-one-GPU rule, three ways:

- **HTCondor** (`hpc/condor/`, the scheduler for Syracuse OrangeGrid — the only
  cluster path in the repo): `+request_gpus = 1` +
  `Requirements = (CUDADriverVersion >= 12.0) && (CUDACapability >= 8.0)` per
  job; Condor sets `CUDA_VISIBLE_DEVICES`, so the scripts' auto-picked `"cuda"`
  IS the assigned card. `marmousi_full_das.sub` queues the 45-combo campaign
  from `combos.txt`; `run.sub` does single acoustic/elastic/field/ladder/matrix
  runs via `-a 'kind=...'`. Edit `hpc/condor/activate_env.sh` once, run
  `fs_check.sub` first. See `hpc/condor/README.md`.
- **Interactive multi-GPU node**: `./hpc/launch_gpus.sh <NGPU>` fans the
  campaign combos across cards in Liu-style batches (combo i on
  cuda:(i mod NGPU), NGPU at a time). `DRY_RUN=1` prints the plan.
- **Manual pinning**: every runner accepts `--device cuda:N` directly.

Memory management per Liu's README: mini-batching (`batch_size`) and
checkpointing (`checkpoint_segments`) — already wired per-misfit with his
exact values in the runners.

## 4e. Cycle-skipping robustness: misfits, ladder, traveltime starter

The autograd does NOT relax FWI's good-starting-model requirement (that is
orthogonal to how the gradient is computed); the misfit + multiscale machinery
does. Three tools for this:

- **9 misfits** now: the 6 original plus `traveltime` (cross-correlation time
  shift) and `nim` (Normalized Integration = Wasserstein-1 at p=1), both
  cycle-skipping robust, plus `convsi` (source-INDEPENDENT convolved-wavefields,
  Choi & Alkhalifah 2011). Available in every runner and the 45-combo campaign.
  `traveltime` is O(shots*receivers) conv1d and slow; decimate. NIM is a
  torch.autograd.Function — custom loops evaluate it via
  `inversion.safe_misfits.apply_misfit`. `convsi` cancels the unknown source
  wavelet + amplitude exactly (run with `waveform_normalize=False`); it is the
  recommended FIELD misfit since the true FORGE source is unknown. `convsi`
  uses FFT -> CUDA/CPU only (no Apple MPS), like `envelope`/`weci`.
- **Starting-model degradation ladder** (`inversion/run_starting_model_ladder.py`):
  inverts the same data from progressively worse starting models (mild ->
  strong smooth -> 1-D gradient -> constant) with each misfit, SGD fixed, and
  reports a recovery heatmap — quantifies each misfit's basin of attraction.
  `--quick` for local wiring; `--misfits gc,envelope,nim,...` to choose the set.
- **First-break traveltime starting model** (`forge/traveltime_tomography.py`):
  STA/LTA first-break picking + VSP check-shot v(z) inversion (straight-ray
  deskew, physical surface anchor) -> a data-driven starting model. Use it in
  the field script with `--starting traveltime` (needs a NEAR-offset shot for a
  reliable profile; a downhole fiber constrains only the depths it spans, so
  the overburden above the fiber top gets the average velocity to that top).

## 4f. One place for all techniques (`inversion/config.py`)

Every technique now lives in a single source of truth, `inversion/config.py`,
instead of being copy-pasted across runners:
- `MISFITS` (9) + `build_misfit(name, dt, iterations)` + `MISFIT_SETTINGS`
- `OPTIMIZER_NAMES` (5) + `LIU_OPTIMIZERS` + `build_optimizer(name, params, lr)`
- `REGULARIZERS` (5) + `build_regularization(name, nx, nz, dx, dz, dev, dtype)`
- `InversionConfig` dataclass (a full technique stack) + `deployment_score`

Every runner (standalone acoustic/elastic/field, the campaign) imports from it;
add a technique in ONE place. Source-independence is the `convsi` misfit;
starting-model construction is `forge/traveltime_tomography.py`; both compose
with any misfit/optimizer/regularizer.

## 4g. Finding the best combination for deployment

The full misfit x optimizer x regularizer x starting-model cross-product is too
large to brute-force. `inversion/run_technique_matrix.py` runs a CONFIGURABLE
grid, scores each run with `deployment_score`, and ranks them. Use it STAGED:

```bash
# 1. base grid: misfit x optimizer (reg=none) -> rank
python inversion/run_technique_matrix.py
# 2. take the top few, sweep regularizers
python inversion/run_technique_matrix.py --misfits gc,sinkhorn --optimizers sgd \
       --regularizers none,tikhonov1,tv1
# 3. take the winner, sweep starting-model quality (the ladder)
python inversion/run_starting_model_ladder.py --misfits gc,sinkhorn
```
`--quick` shrinks it for a local wiring check. Writes `ranking.json` (best
deployment stack first). On OrangeGrid, submit the winning combo with
`condor_submit hpc/condor/run.sub -a 'kind=...' -a 'misfit=...' -a 'optimizer=...'`.

## 5. Syncing the local ADFWI into GitHub

The local (patched) ADFWI package is mirrored at `ADFWI_local/`. After any
change to `../ADFWI/ADFWI/`:

```bash
./scripts/sync_adfwi.sh        # rsync + show what changed
git add ADFWI_local && git commit && git push
```

## 6. Scale warnings

Local runs are DEMOS. Full Marmousi2 reproduction, the FORGE-proxy
campaigns/degradation ladder (`inversion/run_ladder.py`) and field-data
inversion (`inversion/run_field.py`) are HPC work per the build spec
(`../DASFWI_CodeDev_Handoff_v6b.md`, section 3).
