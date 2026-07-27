# Running DASFWI on PSC Bridges-2 (SLURM + H100)

**Bridges-2 is SLURM**, not HTCondor — you submit with `sbatch` and monitor with
`squeue -u $USER`. This `hpc/slurm/` path mirrors `hpc/condor/` one-for-one, and
deliberately **reuses the scheduler-agnostic wrappers** in `hpc/condor/`
(`run_standalone.sh`, `run_combo*.sh`) — only the *scheduler* layer differs. The
science code, the job "kinds", and the combos files are identical across both
clusters.

Workflow is the same as OrangeGrid: **Opus pushes to `Birotimi1/DASFWI`, you
`git pull` on Bridges-2 and run.** Nothing here needs an inbound connection to
the cluster.

One process = one GPU (ADFWI is single-GPU per process). We take **one H100 of a
shared node** via the `GPU-shared` partition — `--gpus=h100-80:1`.

--------------------------------------------------------------------------
## 0. The SU budget — read this first

Allocation **`ees260010p`**: **1,657 GPU SU** and **300 GB Ocean**, through
**2027-04-16**. Bridges-2 charges:

| GPU | SU per GPU-hour |
|-----|-----------------|
| **h100-80** | **2** |
| v100-16 / v100-32 / l40s-48 | 1 |

So the grant is **≈ 828 H100-GPU-hours**. Unlike OrangeGrid (free, opportunistic
GPUs), **every hour here is metered.** The discipline:

1. **Smoke first** (`smoke.sbatch`, ~0.1 SU) — env works on an H100 at all.
2. **Calibrate one run** (`submit.sh calibrate`) — measure wall-clock → SU/run.
3. **Then size the campaign** to fit with margin, and **throttle** the array
   (`%K` in `submit_array.sh`) so the burn rate stays bounded.

`projects` shows live balance; `sacct -j <jobid> --format=JobID,Elapsed,State`
shows what a finished job actually used.

--------------------------------------------------------------------------
## 1. First-time setup (on a LOGIN node — compute nodes have no internet)

```bash
cd /ocean/projects/ees260010p/$USER          # $PROJECT — code+env+data live here
git clone https://github.com/Birotimi1/DASFWI.git
cd DASFWI
bash hpc/slurm/setup_bridges2.sh             # Miniforge + env + torch cu124 + Marmousi
```

`setup_bridges2.sh` is idempotent. It installs Miniforge to
`$PROJECT/miniforge3` (NOT the 25 GB `$HOME`), builds the `dasfwi` env from the
pinned `env.yml`, adds the **torch cu124** wheel (H100 = sm_90; the wheel bundles
its own CUDA runtime, so no `module load cuda` is needed anywhere), downloads
Marmousi2 **here on the login node** (compute nodes can't), and makes
`output/` + `logs/`.

Everything uses defaults that "just work" when the repo is under `/ocean`:
`ADFWI_ROOT`→bundled `ADFWI_local`, `MARMOUSI_DIR`→`../Data_downloads/marmousi2`,
`DASFWI_RESULTS`→`results/marmousi_full_das`.

--------------------------------------------------------------------------
## 2. Submit — single jobs

`submit.sh <kind> [misfit] [optimizer] [-- extra args...]` is the analog of
`condor_submit run.sub -a 'kind=...'`. One H100, per-kind default walltime
(override with `WALLTIME=hh:mm:ss`).

```bash
sbatch hpc/slurm/smoke.sbatch                # 0. env+GPU sanity (~0.1 SU)

hpc/slurm/submit.sh genobs                   # 1. shared obs, ONCE (needs Marmousi)
squeue -u $USER                              #    wait for it

hpc/slurm/submit.sh acoustic gc adam         # a single acoustic DAS run
hpc/slurm/submit.sh acoustic -- --conventional   # the pressure-receiver A/B control
hpc/slurm/submit.sh calibrate                # measure per-run cost / pick rungs
```

kinds: `genobs genobs_elastic calibrate adaptive starter pipeline acoustic
elastic field ladder matrix` (see `hpc/condor/run_standalone.sh`).
Misfits: `l2 envelope gc sdtw sinkhorn weci traveltime nim convsi`.
Optimizers: `sgd adagrad adam adamw nadam`.

--------------------------------------------------------------------------
## 3. Submit — campaigns (job arrays)

`submit_array.sh <combos-file> <wrapper> [max_concurrent] [walltime]` is the
analog of `condor_submit skip_ladder.sub`. `max_concurrent` (`%K`) is your
burn-rate throttle: `K` H100s at once = `2K` SU/wall-hour.

```bash
# the 45-combo acoustic base campaign (2-token lines "misfit optimizer")
hpc/slurm/submit_array.sh hpc/marmousi_full_das/combos.txt \
                          hpc/condor/run_combo.sh 10
```

--------------------------------------------------------------------------
## 4. The adaptive-FWI pipeline (Phases 1-4)

Same phase gates as `ADAPTIVE_FWI_PLAN.md` §5 — Phase 1 first; it sets the λ
schedule's `--flip-lo/--flip-hi` and decides whether the adaptive arm is
justified at all.

```bash
# PHASE 1 — cycle-skip flip test (the hypothesis gate)
hpc/slurm/submit.sh calibrate                          # 1a: pick TRANSITION rungs
cat output/dasfwi_calibrate.*.out
./hpc/marmousi_full_das/make_ladder_combos.sh s16 s20 s24
hpc/slurm/submit_array.sh hpc/marmousi_full_das/combos_ladder.txt \
                          hpc/condor/run_combo_ladder.sh 10        # 135 jobs, <=10 at once
python hpc/marmousi_full_das/flip_curve.py             # FLIP / NO-FLIP verdict -> f_lo,f_hi

# PHASE 2 — adaptive L2->OT objective (three arms), FL/FH from the flip curve
for OBJ in adaptive l2 sinkhorn; do
  hpc/slurm/submit.sh adaptive -- --objective $OBJ --start-rung s20 \
                                  --iters 60 --flip-lo 3 --flip-hi 8
done

# PHASE 3 — Route B transferable starter
hpc/slurm/submit.sh starter -- --iters 80 --band 3.0

# PHASE 4 — full elastic pipeline (+ its control arm)
hpc/slurm/submit.sh genobs_elastic                     # ONCE, elastic obs
hpc/slurm/submit.sh pipeline -- --start route_b --bands 2.0,3.0,4.5,full --iters 50
hpc/slurm/submit.sh pipeline -- --start linear --fixed l2 --bands 2.0,3.0,4.5,full --iters 50
```

Smoke any stage with `--smoke` (2 iters/band, e.g.
`hpc/slurm/submit.sh pipeline -- --smoke`) before the metered run.

--------------------------------------------------------------------------
## 5. Monitor + collect

```bash
squeue -u $USER                                  # queue / running
sacct -X --format=JobID,JobName%20,Elapsed,State,ReqTRES%40 -S today   # what ran + used
tail -f output/dasfwi_<kind>.<jobid>.out         # live log
```

Results land in `$DASFWI_RESULTS/<tag>/` (default
`results/marmousi_full_das/<tag>/`): `iter_vp.npz`, `iter_loss.npz`,
`metrics.json`, `final.png`. Do NOT commit results or SEGY (gitignored) — pull
metrics/plots back with `scp`/DTN or inspect on the cluster.

--------------------------------------------------------------------------
## 6. Differences from OrangeGrid (`hpc/condor/`)

| | OrangeGrid (HTCondor) | Bridges-2 (SLURM) |
|-|-----------------------|-------------------|
| submit | `condor_submit *.sub` | `sbatch` / `submit*.sh` |
| monitor | `condor_q $USER` | `squeue -u $USER` |
| GPU request | `+request_gpus=1` + `Requirements` | `--gpus=h100-80:1 -p GPU-shared` |
| env activation | `hpc/condor/activate_env.sh` | `hpc/slurm/activate_bridges2.sh` (via `DASFWI_ACTIVATE`) |
| conda | Miniforge at `$HOME/miniconda3` | Miniforge at `$PROJECT/miniforge3` (25 GB home) |
| GPU cost | free / opportunistic | **2 SU / H100-hour, metered** |
| preemption | restarts from iter 0 | GPU-shared is **not** opportunistic — no preempt-restart |
| internet on exec node | yes | **no** — stage data/env on the login node |

The dispatch wrappers and combos files are shared; only this directory is
Bridges-2-specific.
