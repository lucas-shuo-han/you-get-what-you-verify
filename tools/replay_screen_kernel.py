"""Replay the exact Kaggle kernel script locally before any push (2-failures rule, DEBUG_LOG 2026-11-19).

Builds the same dataset zip, extracts it into <tmp>/input/brickexp-screen/ the way Kaggle auto-extracts
uploaded zips, rewrites only the /kaggle/{input,working} prefixes and drops the pip line, then runs the
notebook source with the given interpreter (a clean venv with only torch + numpy).
Usage: python tools/replay_screen_kernel.py --python <venv python> --tmp <dir> --run_args "..."
"""
import argparse, json, shutil, subprocess, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_screen as K

ap = argparse.ArgumentParser()
ap.add_argument("--python", required=True)
ap.add_argument("--tmp", required=True)
ap.add_argument("--run_args", required=True)
a = ap.parse_args()
t = Path(a.tmp); shutil.rmtree(t, ignore_errors=True)
(t / "input/brickexp-screen").mkdir(parents=True); (t / "working").mkdir()
d = K.build_zip()
zipfile.ZipFile(d / "screen_code.zip").extractall(t / "input/brickexp-screen")  # Kaggle-style auto-extract
src = K.notebook(a.run_args)["cells"][0]["source"]
assert src.count("pip install -q gurobipy") == 0, "kernel must not need gurobipy"
src = src.replace("/kaggle/input", (t / "input").as_posix()).replace("/kaggle/working", (t / "working").as_posix())
BASH = str(Path(shutil.which("git")).parent.parent / "bin" / "bash.exe")
# Windows shell=True would run "<bash> /c cmd"; emulate Linux semantics: bash -c "<cmd>" (same pipefail as Kaggle)
PRE = """import subprocess as _sp
_run = _sp.run
def _r(cmd, *a, executable=None, shell=False, **k):
    if executable == "/bin/bash":
        return _run([BASH, "-c", cmd], *a, **k)
    return _run(cmd, *a, shell=shell, executable=executable, **k)
_sp.run = _r
"""
src = "BASH = " + repr(BASH) + chr(10) + PRE + src
src = src.replace("python -m brickexp", f'{Path(a.python).as_posix()} -s -E -m brickexp')
(t / "kernel.py").write_text(src)
r = subprocess.run([a.python, str(t / "kernel.py")], cwd=t)
print("replay rc", r.returncode, "| outputs:", sorted(p.name for p in (t / "working").iterdir()))
sys.exit(r.returncode)
