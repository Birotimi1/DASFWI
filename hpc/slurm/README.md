# Slurm submission (GPU)

One process = one GPU (ADFWI is single-GPU per process; the runners auto-pick
`cuda` -> `mps` -> `cpu`, and Slurm sets `CUDA_VISIBLE_DEVICES` so `cuda` is the
allocated card). Mirrors the strategy in `hpc/condor/` for a Slurm cluster.

## One-time setup

1. Edit `hpc/slurm/env.sh` for your cluster (how to load the `dasfwi` conda env
   or set `PYTHON_BIN`; optional path overrides). This is the only runtime-env
   file the jobs source.
2. Set the cluster-specific `#SBATCH` directives (partition / account / qos) —
   either uncomment the `### TODO` lines in each `.slurm`, or pass them on the
   `sbatch` command line (`sbatch --partition=gpu --account=...`).
3. `mkdir -p logs` in the repo root (Slurm writes `logs/%x_%j.out`).
4. GPU request syntax varies: the scripts use `--gres=gpu:1`; some clusters
   need `--gpus=1` or `--gpus-per-task=1` — change it if your scheduler rejects
   `--gres`.

## The 40-combo campaign (array job)

```bash
python hpc/marmousi_full_das/generate_obs.py     # ONCE, before submitting
sbatch hpc/slurm/campaign.slurm                  # array 1-40 -> combos.txt
squeue -u $USER
```
Task `i` runs line `i` of `hpc/marmousi_full_das/combos.txt`; each job's first
stdout line echoes its misfit/optimizer and assigned GPU.

## Single runs (standalone scripts + ladder)

`hpc/slurm/run.slurm` takes parameters via `--export`:

```bash
# acoustic DAS, and the pressure-receiver A/B control:
sbatch --export=ALL,KIND=acoustic,MISFIT=gc,OPT=adam hpc/slurm/run.slurm
sbatch --export=ALL,KIND=acoustic,MISFIT=gc,OPT=adam,EXTRA=--conventional hpc/slurm/run.slurm
# elastic Vp+Vs:
sbatch --export=ALL,KIND=elastic,MISFIT=gc,OPT=adam hpc/slurm/run.slurm
# field 78A-32, production grid, data-driven start:
sbatch --export=ALL,KIND=field,EXTRA="--well 78A-32 --shots 318 --starting traveltime --dz 5 --dt 4e-4 --nt 6000" hpc/slurm/run.slurm
# starting-model degradation ladder:
sbatch --export=ALL,KIND=ladder hpc/slurm/run.slurm
```

Misfits: l2 envelope gc sdtw sinkhorn weci traveltime nim.
Optimizers: sgd adagrad adam adamw nadam.

## Resources

`--mem=24G --cpus-per-task=4 --time=12:00:00` are defaults; sinkhorn/sdtw/
traveltime are slower (they mini-batch / are O(shots*receivers)). Raise
`--time` for the field runs at production resolution.
