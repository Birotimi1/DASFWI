# Running DASFWI on Syracuse OrangeGrid (HTCondor + GPU)

**OrangeGrid is HTCondor** — you submit with `condor_submit` and monitor with
`condor_q $USER` (there is no `sbatch`). This `hpc/condor/` path is the only
scheduler path in the repo. It is self-contained: a fresh reader (or AI model)
can follow it end-to-end.

The submit files and wrappers here follow Syracuse Research Computing's own
OrangeGrid examples (`git clone http://github.com/SyracuseUniversity/OrangeGridExamples`,
see its `Examples/PyTorch`, `Examples/python`, `Examples/CUDA13`,
`Examples/multipleJobs`, `Examples/Checkpointing`). If OrangeGrid changes a
convention, that repo is the source of truth.

One process = one GPU (ADFWI is single-GPU per process; the runners auto-pick
`cuda` and Condor sets `CUDA_VISIBLE_DEVICES` so `cuda` is the assigned card).

--------------------------------------------------------------------------
## 1. The ONLY things to check/adjust

| # | Where | What | How |
|---|-------|------|-----|
| 1 | `hpc/condor/activate_env.sh` | how the conda env is activated on an execute node. **Main thing to get right.** | set `DASFWI_ENV` (default `dasfwi`; use `adfwi` to reuse the torch 2.6+cu124 env you already built). Assumes Miniforge at `$HOME/miniconda3` (§3). |
| 2 | `*.sub` `Requirements` | GPU filter: `(CUDADriverVersion >= 12.0) && (CUDACapability >= 8.0)`. | leave as-is for the fast Ampere cards; drop the `CUDACapability` clause to also match the Turing Quadros (§ GPU). |
| 3 | `*.sub` `request_memory` | 24G campaign / 48G single runs. | raise for field production. |
| 4 | shared filesystem | do execute nodes see the repo + data? OrangeGrid is opportunistic. | run `fs_check.sub` FIRST (§4). |
| 5 | `DASFWI_RESULTS` (optional) | where outputs go. | `-a 'environment=DASFWI_RESULTS=/path'` or export before submit. |

Already set in every `.sub`: `+request_gpus = 1` (OrangeGrid's leading-plus
form), `should_transfer_files = IF_NEEDED`, `initialdir = .` (the repo root),
`.out`/`.err` under `output/`, `.log` under `logs/`. The wrappers activate conda
themselves, so **`getenv` is not used** (it can leak a stale PATH over Miniforge).

--------------------------------------------------------------------------
## 2. First-time setup

```bash
mkdir -p ~/das-fwi && cd ~/das-fwi
git clone https://github.com/Birotimi1/DASFWI.git   # brings ADFWI_local/ + all runners
# copy the data folders NEXT TO DASFWI/ (not in git):
#   Data_downloads/marmousi2/*.segy.gz   (campaign; can also auto-download)
#   DAS_VSP/{78A-32,78B-32}/*.sgy         (field runs)
cd DASFWI && mkdir -p output logs        # condor needs both dirs to exist
```

--------------------------------------------------------------------------
## 3. Environment (Miniforge, the OrangeGrid way)

OrangeGrid runs Conda from **Miniforge installed at `$HOME/miniconda3`**, and
jobs enter it with `eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash
hook)"; conda activate <env>` — exactly what `activate_env.sh` does. Install it
once (from the OrangeGrid PyTorch example):

```bash
wget https://github.com/conda-forge/miniforge/releases/download/24.7.1-0/Miniforge-pypy3-24.7.1-0-Linux-x86_64.sh
bash Miniforge-pypy3-24.7.1-0-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
conda init
# add to ~/.bash_profile so logins pick it up:
#   if [ -e ${HOME}/.bashrc ]; then source ${HOME}/.bashrc; fi
```

Then build the DASFWI env. The code was developed on the pinned stack in
`env.yml` (python 3.10, numpy 1.24.4, scipy 1.10.1, obspy, pysdtw, geomloss,
POT). Those non-torch pins matter; torch/CUDA is the flexible part.

- **Recreate `dasfwi` from `env.yml`**, changing only the torch line to
  OrangeGrid's CUDA (cu124 works on its drivers):
  ```bash
  mamba env create -f env.yml            # or: conda env create -f env.yml
  conda activate dasfwi
  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
  ```
- **Or reuse the `adfwi` env** you already built (python 3.11 / torch 2.6+cu124):
  set `DASFWI_ENV=adfwi` in `activate_env.sh` and `pip install` any missing deps
  (`pysdtw geomloss POT segyio 'setuptools<81'`). The code runs on torch 2.1–2.6.

Verify on a submit node:
`python -c "import ADFWI, das, forge, inversion, torch; print(torch.__version__, torch.cuda.is_available())"`.

--------------------------------------------------------------------------
## 4. Pre-flight (verify shared FS + GPU on an execute node)

```bash
condor_submit hpc/condor/fs_check.sub
condor_q $USER                        # wait for it to finish
cat output/fs_check_*.out             # repo/data visible? conda env OK? GPU?
```
Any `MISSING` line means that path is not visible where jobs land — stage it
(`should_transfer_files`) or use a shared scratch, and set `MARMOUSI_DIR` /
`DASFWI_RESULTS` accordingly.

--------------------------------------------------------------------------
## GPU selection

OrangeGrid mixes GPU generations. `condor_status` / the diagnostics example show
the pool as **NVIDIA A100 80GB PCIe, NVIDIA A40, NVIDIA L40S** (Ampere/Ada,
capability ≥ 8.0) plus older **Quadro RTX 6000/5000** (Turing, capability 7.5).

The submit files request:
```
+request_gpus = 1
Requirements  = (CUDADriverVersion >= 12.0) && (CUDACapability >= 8.0)
```
- `CUDADriverVersion >= 12.0` — a CUDA-12 driver, so a torch **cu124** build runs
  (this is the line OrangeGrid's PyTorch example uses).
- `CUDACapability >= 8.0` — pins to the fast A100 80GB / A40 / L40S and skips the
  Turing Quadros. **Drop this clause** to widen the pool (the Quadros also run
  torch) if queue waits are long.
- To pin one model: `&& (CUDADeviceName == "NVIDIA A100 80GB PCIe")`.

(If a future OrangeGrid rejects `+request_gpus`, its examples also document
`require_gpus = (Capability >= 8.0)` — verify the attribute name in the current
`Examples/CUDA13` / GPU readme.)

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

condor_q $USER ; condor_tail <job.id>       # monitor
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

--------------------------------------------------------------------------
## 7. Preemption (opportunistic pool)

OrangeGrid nodes can vanish (reboot, network, hardware) and HTCondor then
**restarts the job from the beginning** on another node (see the OrangeGrid
Checkpointing example). The runners cache `iter_vp.npz` every 10 iterations, but
a restarted job currently re-starts at iteration 0. For short test runs this is
fine. For the long full campaign, either accept the occasional restart or add
application-level resume (load the newest cached `iter_vp` on startup) — a
tracked follow-up, not required for the first test submission.
