"""Can this file's first-party imports RESOLVE in a bare shell?

Written after a sweep that used `--help` reported everything clean and then
FAILED to flag the file we had just watched crash on the cluster: argparse
exits on --help before main() runs, and these drivers import inside main().
A detector that cannot catch the known bug proves nothing, so this one is
validated against that exact file before being trusted.

Executes the module with __name__ != "__main__" (running its sys.path
bootstrap and module-level imports, but not its main()), then walks the AST for
every first-party import ANYWHERE -- including inside function bodies -- and
tries each one.
"""
import ast
import importlib
import os
import subprocess
import sys
import traceback
from pathlib import Path
from pathlib import Path as _Path

FIRST_PARTY = ("ADFWI", "forge", "inversion", "hpc")


def first_party_imports(tree):
    """Every first-party module named by an import anywhere in the tree."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in FIRST_PARTY:
                    out.add(a.name)
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            if n.module.split(".")[0] in FIRST_PARTY:
                out.add(n.module)
    return sorted(out)


def check(path):
    path = Path(path).resolve()
    src = path.read_text()
    tree = ast.parse(src)
    g = {"__name__": "__not_main__", "__file__": str(path)}
    try:
        exec(compile(src, str(path), "exec"), g)      # runs the bootstrap
    except Exception:
        return f"module-level exec failed:\n{traceback.format_exc(limit=3)}"
    bad = []
    for m in first_party_imports(tree):
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as e:
            bad.append(f"{m}: {e}")
        except Exception:
            pass          # import worked; failure is downstream, not our concern
    return "; ".join(bad)


def sweep(root):
    """Every runnable entry point in the repo. Run from anywhere, no env."""
    # rglob, NOT glob("*/*.py"): hpc/standalone/ is TWO levels down, so the
    # shallow pattern silently skipped all seven drivers and still printed
    # "all entry points resolve". A sweep must report its own coverage, or
    # checking nothing looks exactly like checking everything.
    bad = n = 0
    skip = {"ADFWI_local", ".git", "results", "output"}
    for f in sorted(root.rglob("*.py")):
        if skip & set(f.relative_to(root).parts):
            continue
        if "__main__" not in f.read_text(errors="replace"):
            continue
        n += 1
        # SUBPROCESS per file. Exec'ing every module into one interpreter made
        # modules contaminate each other: run_traveltime_starter.py reported
        # "exec failed" in the sweep and OK on its own. A guard that cries wolf
        # gets ignored, so each file is checked in isolation, exactly as the
        # single-file mode that was validated against the known-broken file.
        r = subprocess.run([sys.executable, __file__, str(f)],
                           capture_output=True, text=True, timeout=300,
                           env={k: v for k, v in os.environ.items()
                                if k != "PYTHONPATH"})
        problem = "" if r.returncode == 0 else (r.stdout + r.stderr).strip()
        if problem:
            bad += 1
            # the LAST line is the exception; the first is just our banner
            print(f"  BROKEN  {f.relative_to(root)}  ->  "
                  f"{problem.splitlines()[-1][:110]}")
    print(f"{n - bad}/{n} entry points resolve imports in a bare shell")
    if n == 0:
        print("  *** checked NOTHING -- the glob is wrong, not the code ***")
        return 1
    return bad


if __name__ == "__main__":
    if len(sys.argv) > 1:
        problem = check(sys.argv[1])
        print(problem if problem else "OK")
        sys.exit(1 if problem else 0)
    # no args: sweep the repo. Subprocess-free so one bad module cannot
    # poison the rest -- each check() re-imports into a fresh namespace.
    root = _Path(__file__).resolve().parents[1]
    sys.exit(1 if sweep(root) else 0)
