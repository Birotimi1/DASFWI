# Running DASFWI on Syracuse OrangeGrid (Slurm + A6000 GPU)

This guide is self-contained: a person (or a fresh AI model) with no prior
context can follow it end-to-end. It tells you **exactly which paths to check
and adjust**, what each one means, and how to submit. One process = one GPU
(ADFWI is single-GPU per process; the runners auto-pick `cuda` -> `mps` ->
`cpu`, and Slurm sets `CUDA_VISIBLE_DEVICES` so `cuda` is the allocated card).

--------------------------------------------------------------------------
## 1. Directory layout — what lives where

The code is one git repo; two data folders live OUTSIDE it and are NOT in git
(too big / not public). The recommended layout on the cluster:

```
$HOME/ (or /scratch/$USER/) 
└── Codes/                              # any parent dir; call it what you like
    ├── DASFWI/                         # THE REPO: git clone Birotimi1/DASFWI
    │   ├── das/  forge/  inversion/    # the package
    │   ├── ADFWI_local/                # the ADFWI engine, BUNDLED in the repo
    │   │   └── ADFWI/                  #   (no separate ADFWI clone needed)
    │   ├── hpc/slurm/                  # <- you are here
    │   ├── tests/                      # pytest suite
    │   └── env.yml                     # conda environment spec
    ├── Data_downloads/
    │   └── marmousi2/                  # Marmousi2 SEGY (for the campaign);
    │       ├── vp_marmousi-ii.segy.gz  #   copy from local OR let it auto-
    │       ├── vs_marmousi-ii.segy.gz  #   download on first use
    │       └── density_marmousi-ii.segy.gz
    └── DAS_VSP/                        # FORGE field data (for run KIND=field)
        ├── 78A-32/*.sgy                #   318 walkaway shots, well 78A-32
        └── 78B-32/*.sgy                #   318 walkaway shots, well 78B-32
```

Why side-by-side matters: the code finds the two data folders RELATIVE to the
repo — it looks for `../Data_downloads/marmousi2` and `../DAS_VSP` (i.e. one
level above `DASFWI/`). Keep that layout and you set zero paths. If you can't,
override them with environment variables (step 3).

--------------------------------------------------------------------------
## 2. The ONLY paths/values you must check or adjust

Everything else is already wired. Adjust these:

| # | Where | What | How |
|---|-------|------|-----|
| 1 | `hpc/slurm/env.sh` | How to load the `dasfwi` conda env (or set `PYTHON_BIN`). **This is the main thing to get right on a new cluster.** | Edit the file; see step 4. |
| 2 | `hpc/slurm/*.slurm` `--constraint` | GPU type string. It's set to `gpu_type:A6000` but the exact token varies per cluster. | Run `sinfo -o "%f"` on OrangeGrid and match the A6000 feature name; fix if different. |
| 3 | `hpc/slurm/*.slurm` `--mail-user` | Already `Birotimi@syr.edu`. | Change if a different address should get job mail. |
| 4 | `DASFWI_RESULTS` (optional) | Where outputs are written. Default: `DASFWI/results/`. On a cluster you usually want scratch. | `export DASFWI_RESULTS=/scratch/$USER/dasfwi_results` (in `env.sh` or your shell). |
| 5 | `MARMOUSI_DIR`, `ADFWI_ROOT` (optional) | Only if the side-by-side layout (§1) is NOT used. | Set in `env.sh`. Defaults resolve automatically when side-by-side. |
| 6 | `--account` (maybe) | OrangeGrid's example headers show no account. | Add `#SBATCH --account=...` in the `.slurm` files only if submission is rejected for a missing allocation. |

Things already set for OrangeGrid (no action): `--partition=gpu_zone2,gpu`,
`--gres=gpu:1`, `--cpus-per-task=10`, `--nodes=1`, `--ntasks-per-node=1`,
`--mail-type=ALL`, `--requeue`, log files under `logs/`.

--------------------------------------------------------------------------
## 3. First-time setup (once)

```bash
# a) get the code (self-contained: ADFWI is bundled in ADFWI_local/)
cd /scratch/$USER          # or wherever; keep the §1 layout
mkdir -p Codes && cd Codes
git clone https://github.com/Birotimi1/DASFWI.git

# b) copy the data folders next to DASFWI/ (NOT in git). From your local machine:
#    rsync -av <local>/Data_downloads/marmousi2   $USER@orangegrid:/scratch/$USER/Codes/Data_downloads/
#    rsync -av <local>/DAS_VSP                     $USER@orangegrid:/scratch/$USER/Codes/DAS_VSP/
#    (Marmousi can instead auto-download on first use if the node has internet.)

# c) build the conda env
cd DASFWI
conda env create -f env.yml          # creates env "dasfwi"
#    if env.yml resolution is slow, the explicit pins are in HPC notes; the
#    essentials: python=3.10, torch (CUDA build matching the cluster), numpy
#    scipy matplotlib obspy 'setuptools<81' segyio pysdtw geomloss POT numba
#    -> IMPORTANT: install the CUDA torch build, e.g.
#       pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

--------------------------------------------------------------------------
## 4. Edit `hpc/slurm/env.sh` for OrangeGrid (the key step)

`env.sh` is sourced by every job and must make `python` be the `dasfwi` env's
interpreter on a compute node. Two common ways — use whichever works on
OrangeGrid:

```bash
# Option A: module + conda activate
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dasfwi
PYTHON_BIN=python

# Option B: absolute interpreter path (no activation, most robust)
PYTHON_BIN=/scratch/$USER/miniconda3/envs/dasfwi/bin/python
```

Verify interactively on a GPU node before submitting:
```bash
srun --partition=gpu_zone2,gpu --constraint=gpu_type:A6000 --gres=gpu:1 \
     --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
source hpc/slurm/env.sh          # should print: torch ... cuda_available True ...
python -c "import ADFWI, das, forge, inversion; print('imports OK')"
```

--------------------------------------------------------------------------
## 5. Submit

```bash
cd DASFWI
mkdir -p logs                                    # Slurm writes logs/ here

# --- the observable question (do these FIRST): DAS vs conventional A/B ---
sbatch --export=ALL,KIND=acoustic,MISFIT=gc,OPT=adam hpc/slurm/run.slurm
sbatch --export=ALL,KIND=acoustic,MISFIT=gc,OPT=adam,EXTRA=--conventional hpc/slurm/run.slurm

# --- the full 40-combo campaign (array job; generate shared data ONCE) ---
python hpc/marmousi_full_das/generate_obs.py     # writes the observed strain rate
sbatch hpc/slurm/campaign.slurm                  # 40 tasks, one A6000 each

# --- elastic Vp+Vs ---
sbatch --export=ALL,KIND=elastic,MISFIT=gc,OPT=adam hpc/slurm/run.slurm

# --- FORGE field data, production grid, data-driven starting model ---
sbatch --export=ALL,KIND=field,EXTRA="--well 78A-32 --shots 318 --starting traveltime --dz 5 --dt 4e-4 --nt 6000" hpc/slurm/run.slurm

# --- starting-model degradation ladder ---
sbatch --export=ALL,KIND=ladder hpc/slurm/run.slurm

# monitor
squeue -u $USER
tail -f logs/dasfwi-campaign_*_1.out             # a task's live output
```

Misfits: `l2 envelope gc sdtw sinkhorn weci traveltime nim`.
Optimizers: `sgd adagrad adam adamw nadam`.

--------------------------------------------------------------------------
## 6. Verify a run worked / collect results

Each run writes into `$DASFWI_RESULTS/<tag>/` (default `results/<tag>/`):
`iter_vp.npz`, `iter_loss.npz`, `metrics.json` (RMS init/final, update
correlation, runtime; the campaign also has near/far-fibre splits), `final.png`.
The array campaign writes `results/marmousi_full_das/<misfit>_<optimizer>/`.

```bash
# quick pass/fail across the campaign:
for d in results/marmousi_full_das/*/; do
  python -c "import json,sys; m=json.load(open('$d/metrics.json'));
print('$d', 'RMS', round(m['rms_init']),'->',round(m['rms_final']),
      'corr', round(m['update_corr'],3))" 2>/dev/null
done
```

A job's first stdout line prints its host, misfit/optimizer, and
`CUDA_VISIBLE_DEVICES` — confirm it is NOT `unset` (that would mean no GPU was
allocated). `nvidia-smi` inside a job shows the A6000.

--------------------------------------------------------------------------
## 7. Notes

- One process, one GPU: this is deliberate (ADFWI is single-GPU; `gpu_num` is
  inert upstream). Request one GPU per job; the 40-combo array fans across the
  pool automatically.
- Runtime: campaign combos ~1-4 h on an A6000; `sinkhorn`/`sdtw`/`traveltime`
  are slower (mini-batched / O(shots*receivers)). Field production is the
  largest — `--time=2-00:00:00` in `run.slurm`.
- Do NOT commit results or the SEGY data (both are gitignored). Push only code
  and small summaries.
