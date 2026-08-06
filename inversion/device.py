"""Device selection that REFUSES to silently spend a GPU allocation on CPU.

Every driver had the same fallback:

    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

Sensible on a laptop. On Bridges-2 it is the most expensive failure available.
If a job holds an H100 and CUDA is not visible -- wrong module, driver
mismatch, a node that hands back no card -- the run does not stop. It quietly
computes on the job's few CPU cores at a small fraction of H100 throughput,
bills 2 SU/hour for the full 8-hour walltime, and ends in a timeout or a
partial model. Nothing in the log says "no GPU"; it just looks slow.

With the allocation nearly spent, one such job costs more than every bug found
so far. So: under SLURM, with a GPU requested, absence of CUDA is a HARD
FAILURE at startup, before any SU is burned on it.

An explicit --device still wins, because deliberately running on CPU is a
legitimate thing to ask for. The refusal is only for the case where nobody
chose CPU and nobody would notice.
"""
import os


def slurm_gpu_requested():
    """Are we inside a SLURM job that asked for a GPU?

    Checked against several variables because sites differ in which they set:
    Bridges-2 sets SLURM_JOB_GPUS on GPU-shared, others only export
    CUDA_VISIBLE_DEVICES. Any one of them means a card was requested.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        return False
    for v in ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE",
              "SLURM_STEP_GPUS", "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(v, "")
        # "NoDevFiles" COUNTS as requested. SLURM only writes it when GPUs were
        # asked for and the gres device files could not be handed over -- i.e.
        # precisely the broken-card case this guard exists to catch. Excluding
        # it as "no GPU" let the worst scenario fall straight through to CPU.
        if val and val != "-1":
            return True
    return False


def pick_device(arg=None, allow_cpu=False):
    """'cuda' / 'mps' / 'cpu', refusing a silent CPU fallback inside a GPU job.

    `arg`       explicit user choice; honoured verbatim, guard skipped.
    `allow_cpu` set True for tools that legitimately run CPU-only under SLURM
                (a preflight, a plotter), so they are not caught by the guard.
    """
    import torch

    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    # The guard sits ABOVE the mps fallback, not below it. Inside a SLURM job
    # that holds a card, ANY fallback is wrong -- not just the CPU one -- and
    # putting mps first made the guard untestable on a Mac, which is where it
    # gets developed. It is unreachable code on Bridges-2 either way; ordering
    # it correctly is what let this be tested at all.
    if slurm_gpu_requested() and not allow_cpu:
        raise RuntimeError(
            "This job holds a GPU but torch.cuda.is_available() is False, so "
            "the run would proceed on CPU: many times slower, billing the full "
            "walltime, and almost certainly ending in a timeout.\n"
            f"  SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')}  "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}\n"
            f"  torch {torch.__version__}, built for CUDA "
            f"{getattr(torch.version, 'cuda', None)!r}\n"
            "Check the CUDA module and the torch build. To run on CPU on "
            "purpose, pass --device cpu."
        )
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
