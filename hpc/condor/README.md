# Running DASFWI on Syracuse OrangeGrid (HTCondor + GPU)

**OrangeGrid is HTCondor** (confirmed: `condor_submit`/`condor_q`, no `sbatch`).
This `hpc/condor/` path is the LIVE one for OrangeGrid. The `hpc/slurm/` scripts
are for a Slurm cluster and are NOT used here. Self-contained: a fresh reader
(or AI model) can follow this end-to-end.

One process = one GPU (ADFWI is single-GPU per process; the runners auto-pick
`cuda` and Condor sets `CUDA_VISIBLE_DEVICES` so `cuda` is the assigned card).

--------------------------------------------------------------------------
## 1. The ONLY things to check/adjust

| # | Where | What | How |
|---|-------|------|-----|
| 1 | `hpc/condor/activate_env.sh` | how the conda env is activated on an execute node (Miniforge on OrangeGrid). **Main thing to get right.** | edit `DASFWI_ENV` (default `dasfwi`; `adfwi` to reuse your existing env — see §3). |
| 2 | `*.sub` `requirements` | GPU capability filter, set to `CUDACapability >= 8.0` (A40 & A6000 are 8.6). | if Condor rejects it, use `require_gpus = (Capability >= 8.0)`; verify the attribute name in OrangeGrid's GPU readme. |
| 3 | `*.sub` `request_memory` | 24G campaign / 48G single runs. | raise for field production. |
| 4 | shared filesystem | do execute nodes see the repo + data? OrangeGrid is opportunistic. | run `fs_check.sub` FIRST (§4). |
| 5 | `DASFWI_RESULTS` (optional) | where outputs go. | `-a 'environment=DASFWI_RESULTS=/path'` or export before submit. |

Already set: `request_gpus = 1`, `getenv = True`, `should_transfer_files =
IF_NEEDED`, `initialdir = .` (the repo root), logs under `logs/`.

--------------------------------------------------------------------------
## 2. First-time setup

```bash
mkdir -p ~/das-fwi && cd ~/das-fwi
git clone https://github.com/Birotimi1/DASFWI.git   # brings ADFWI_local/ + all runners
# copy the data folders NEXT TO DASFWI/ (not in git):
#   Data_downloads/marmousi2/*.segy.gz   (campaign; can also auto-download)
#   DAS_VSP/{78A-32,78B-32}/*.sgy         (field runs)
cd DASFWI && mkdir -p logs
```

--------------------------------------------------------------------------
## 3. Environment (Miniforge)

The code was developed on the pinned stack in `env.yml` (python 3.10,
numpy 1.24.4, scipy 1.10.1, obspy, pysdtw, geomloss, POT). Those non-torch
pins matter (the misfit code depends on them). torch/CUDA is the flexible part
— use a build matching OrangeGrid's drivers.

Two options:
- **Recreate `dasfwi` from `env.yml`** (reproducible with local work), changing
  only the torch line to the cluster's CUDA (cu124 works on OrangeGrid):
  ```bash
  mamba env create -f env.yml            # or conda
  mamba activate dasfwi
  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
  ```
- **Reuse an existing env** (e.g. the `adfwi` env you already built with
  python 3.11 / torch 2.6+cu124): set `DASFWI_ENV=adfwi` in `activate_env.sh`
  and `pip install` any missing deps (pysdtw geomloss POT segyio 'setuptools<81').
  The code runs on torch 2.1–2.6; the pinned numpy/scipy still matter.

Verify interactively on a submit node: `python -c "import ADFWI, das, forge,
inversion, torch; print(torch.__version__, torch.cuda.is_available())"`.

--------------------------------------------------------------------------
## 4. Pre-flight (verify shared FS + GPU on an execute node)

```bash
condor_submit hpc/condor/fs_check.sub
condor_q                              # wait for it to finish
cat logs/fs_check_*.out               # repo/data visible? conda env OK? GPU?
```
Any `MISSING` line means that path is not visible where jobs land — stage it
(`should_transfer_files`) or use a shared scratch, and set `MARMOUSI_DIR` /
`DASFWI_RESULTS` accordingly.

--------------------------------------------------------------------------
## 5. Submit

```bash
python hpc/marmousi_full_das/generate_obs.py     # ONCE, before the campaign

# the observable question first: DAS vs conventional A/B
condor_submit hpc/condor/run.sub -a 'kind=acoustic' -a 'misfit=gc' -a 'optimizer=adam'
condor_submit hpc/condor/run.sub -a 'kind=acoustic' -a 'extra=--conventional'

# the full 45-combo campaign (9 misfits x 5 optimizers, one GPU each):
condor_submit hpc/condor/marmousi_full_das.sub

# elastic Vp+Vs; FORGE field (source-independent misfit, data-driven start):
condor_submit hpc/condor/run.sub -a 'kind=elastic' -a 'misfit=gc' -a 'optimizer=adam'
condor_submit hpc/condor/run.sub -a 'kind=field' -a 'misfit=convsi' \
    -a 'extra=--well 78A-32 --shots 318 --starting traveltime --dz 5 --dt 4e-4 --nt 6000'

# technique-matrix search / degradation ladder:
condor_submit hpc/condor/run.sub -a 'kind=matrix'
condor_submit hpc/condor/run.sub -a 'kind=ladder' -a 'extra=--misfits gc,sinkhorn'

condor_q ; condor_tail <job.id>       # monitor
```
Misfits: l2 envelope gc sdtw sinkhorn weci traveltime nim convsi.
Optimizers: sgd adagrad adam adamw nadam.

--------------------------------------------------------------------------
## 6. Collect results

Each run writes `$DASFWI_RESULTS/<tag>/` (default `results/<tag>/`):
`iter_vp.npz`, `iter_loss.npz`, `metrics.json`, `final.png`; the campaign writes
`results/marmousi_full_das/<misfit>_<optimizer>/`. A job's first stdout line
prints host / misfit / optimizer / `CUDA_VISIBLE_DEVICES` — confirm it is NOT
`unset`. Do NOT commit results or SEGY data (gitignored).
