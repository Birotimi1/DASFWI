#!/usr/bin/env bash
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --job-name=dasfwi_smoke
#SBATCH --output=output/%x.%j.out
#SBATCH --error=output/%x.%j.err
# ---------------------------------------------------------------------------
# Near-free H100 validation (~0.1-0.2 SU): confirms the env activates ON A GPU
# NODE, torch sees the H100, ADFWI + inversion import, and a real GPU op runs.
# Needs NO Marmousi/obs -- run it right after setup_bridges2.sh and BEFORE
# spending SU on genobs + the campaign.
#
#   sbatch hpc/slurm/smoke.sbatch
#
# For a full 2-iteration FWI smoke (that one needs obs, so run genobs first):
#   hpc/slurm/submit.sh adaptive -- --smoke
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export DASFWI_ACTIVATE="hpc/slurm/activate_bridges2.sh"
source "$DASFWI_ACTIVATE"          # prints python / torch / cuda / device

echo "=== H100 GPU op + import check on $(hostname) ==="
"$PYTHON_BIN" - <<'PY'
import sys; sys.path.insert(0, "ADFWI_local")
import torch, ADFWI, inversion
assert torch.cuda.is_available(), "no CUDA device visible inside the job!"
dev = torch.device("cuda")
x = torch.randn(4096, 4096, device=dev)
y = (x @ x).sum().item()           # a real H100 matmul
print(f"OK: {torch.cuda.get_device_name(0)} | matmul sum={y:.3e} | "
      f"ADFWI + inversion import clean")
PY
echo "=== smoke passed -- env is good, safe to spend SU ==="
