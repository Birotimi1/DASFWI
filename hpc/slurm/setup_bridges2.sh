#!/usr/bin/env bash
# ============================================================================
# ONE-TIME PSC Bridges-2 setup for DASFWI. RUN THIS ON A LOGIN NODE.
#
#   Bridges-2 COMPUTE nodes have NO outbound internet -- so the conda env, the
#   torch wheel, and the Marmousi download must ALL happen here, on the login
#   node, before any sbatch job. A genobs GPU job that tried to wget Marmousi
#   would fail AND burn H100 SU. This script front-loads every network step.
#
# Idempotent: re-running skips whatever is already in place.
#
#   cd /ocean/projects/ees260010p/$USER
#   git clone https://github.com/Birotimi1/DASFWI.git      # if not already
#   cd DASFWI
#   bash hpc/slurm/setup_bridges2.sh
#
# Overrides: DASFWI_CONDA_ROOT, DASFWI_ENV, MARMOUSI_DIR, PROJECT.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

PROJECT="${PROJECT:-/ocean/projects/ees260010p/$(whoami)}"
CONDA_ROOT="${DASFWI_CONDA_ROOT:-$PROJECT/miniforge3}"
ENV_NAME="${DASFWI_ENV:-dasfwi}"
MARMOUSI_DIR="${MARMOUSI_DIR:-$(dirname "$REPO")/Data_downloads/marmousi2}"

echo "=== DASFWI Bridges-2 setup ==="
echo "  repo        $REPO"
echo "  miniforge   $CONDA_ROOT"
echo "  env         $ENV_NAME"
echo "  marmousi    $MARMOUSI_DIR"
echo "  host        $(hostname)"

# $HOME is only 25 GB on Bridges-2; the env alone is several GB. Refuse to build
# a conda root on home -- push the user to Ocean.
case "$CONDA_ROOT" in
  /ocean/*) : ;;
  *) echo "!! CONDA_ROOT ($CONDA_ROOT) is not under /ocean. \$HOME is only 25 GB;" >&2
     echo "!! set DASFWI_CONDA_ROOT to a path in $PROJECT and re-run." >&2
     exit 2 ;;
esac
case "$REPO" in
  /ocean/*) : ;;
  *) echo "   WARNING: repo is at $REPO (not /ocean). Results/checkpoints can" >&2
     echo "   overflow the 25 GB home -- consider cloning into $PROJECT." >&2 ;;
esac

# ---- 1. Miniforge (conda-forge base = the validated stack, no PSC anaconda) --
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    echo ">>> [1/5] installing Miniforge -> $CONDA_ROOT"
    tmp="$(mktemp -d)"
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
        -O "$tmp/miniforge.sh"
    bash "$tmp/miniforge.sh" -b -p "$CONDA_ROOT"
    rm -rf "$tmp"
else
    echo ">>> [1/5] Miniforge present -> $CONDA_ROOT"
fi
eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"

# ---- 2. the dasfwi env from the pinned env.yml ------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> [2/5] env '$ENV_NAME' exists -- updating from env.yml"
    conda env update -n "$ENV_NAME" -f env.yml
else
    echo ">>> [2/5] creating env '$ENV_NAME' from env.yml"
    conda env create -n "$ENV_NAME" -f env.yml
fi
conda activate "$ENV_NAME"

# ---- 3. torch cu124 (H100 = sm_90; the cu124 wheel bundles its CUDA runtime) -
if python -c "import torch" 2>/dev/null; then
    echo ">>> [3/5] torch already installed ($(python -c 'import torch;print(torch.__version__)'))"
else
    echo ">>> [3/5] installing torch (cu124)"
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
fi

# ---- 4. Marmousi2 SEGY (download HERE; compute nodes cannot) -----------------
echo ">>> [4/5] Marmousi2 -> $MARMOUSI_DIR"
mkdir -p "$MARMOUSI_DIR"
missing=0
for fn in vp_marmousi-ii.segy.gz vs_marmousi-ii.segy.gz density_marmousi-ii.segy.gz; do
    if [ -s "$MARMOUSI_DIR/$fn" ]; then
        echo "    have $fn"
    else
        echo "    fetching $fn ..."
        wget -q "http://www.agl.uh.edu/downloads/$fn" -P "$MARMOUSI_DIR" \
            && echo "    got  $fn" || { echo "    FAILED $fn"; missing=1; }
    fi
done
if [ "$missing" = 1 ]; then
    echo "    !! some Marmousi files did not download (agl.uh.edu may be down)."
    echo "    !! stage them via the DTN (data.bridges2.psc.edu) or Globus into"
    echo "    !! $MARMOUSI_DIR before submitting the genobs job."
fi

# ---- 5. run dirs + import sanity (login node has NO GPU -> cuda False is OK) --
mkdir -p "$REPO/output" "$REPO/logs"
echo ">>> [5/5] import sanity (cuda=False on the login node is EXPECTED)"
python - <<PY
import sys; sys.path.insert(0, "ADFWI_local")
import torch, ADFWI, inversion
print(f"    OK: torch {torch.__version__}, ADFWI + inversion import clean")
print(f"    (GPU visible here: {torch.cuda.is_available()} -- real check is the smoke sbatch)")
PY

echo ""
echo "=== setup complete ==="
echo "next: a near-free GPU smoke, then the pipeline (see hpc/slurm/README.md):"
echo "  sbatch hpc/slurm/smoke.sbatch"
echo "  hpc/slurm/submit.sh genobs        # generate shared obs (ONE H100 job)"
