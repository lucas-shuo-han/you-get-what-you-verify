"""One idempotent "tick" of the unattended long-training chain for group A (run every 30 min by Task Scheduler).

State machine (local state in results/long_A/):
  kernel never pushed               -> push session 1
  kernel queued/running             -> nothing
  kernel complete / error / cancel  -> download its output to results/long_A/sessions/<n>/
       progress.log has "ALL DONE"  -> copy to results/long_A/final/, write DONE, stop forever
       session failed (error.txt or kernel error) twice in a row -> write STOP_NEEDS_DEBUG, stop (no blind resubmits)
       otherwise                    -> upload the output as a new version of dataset <user>/brickexp-long-state
                                       and push the next session, which restores that state and resumes
  push refused (e.g. weekly GPU quota) -> log it, retry on the next tick
Credentials: the kaggle CLI's own config (~/.kaggle); nothing secret is read or printed here.
Usage: python tools/kaggle_long_tick.py [--dry]
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import kaggle_push as K  # noqa: E402  (sh(), username(), token env handling)

D = ROOT / "results/long_A"
D.mkdir(parents=True, exist_ok=True)
STATE_F, LOG_F = D / "tick_state.json", D / "tick.log"
KSLUG, SSLUG = "brickexp-long-a", "brickexp-long-state"
DRY = "--dry" in sys.argv


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_F, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status_md(msg):
    (D / "STATUS.md").write_text(f"# Long training A: status\n\nUpdated {time.ctime()}\n\n{msg}\n", encoding="utf-8")


def kaggle(*a):
    rc, out = K.sh(*a, check=False)
    return rc, out


def push_kernel(user, with_state):
    nb = ROOT / "notebooks/kaggle_long_A.ipynb"
    subprocess.run([sys.executable, str(ROOT / "tools/build_colab_long_notebook.py"), "--kaggle"], check=True)
    k = K.STAGE / "kernel_long_A"
    shutil.rmtree(k, ignore_errors=True); k.mkdir(parents=True)
    shutil.copy(nb, k / "run.ipynb")
    ds = [f"{user}/brickexp-code"] + ([f"{user}/{SSLUG}"] if with_state else [])
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KSLUG}", "title": KSLUG, "code_file": "run.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ds, "competition_sources": [], "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=1))
    if DRY:
        log(f"DRY: would push kernel with datasets {ds}"); return True
    rc, out = kaggle("kernels", "push", "-p", str(k), "--accelerator", "NvidiaTeslaT4", "-t", str(12 * 3600))
    ok = rc == 0 and "error" not in out.lower()
    log(("pushed kernel session " if ok else "PUSH REFUSED: " + out.strip()[-300:] + " | session ") + str(with_state))
    return ok


def upload_state(user, src: Path):
    stage = K.STAGE / "long_state"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(src, stage, ignore=shutil.ignore_patterns("merged_tmp", "probe", "*.ipynb", "__results__*", "__notebook__*",
                                                              "__output__*", "custom.css"))
    (stage / "dataset-metadata.json").write_text(json.dumps({"title": SSLUG, "id": f"{user}/{SSLUG}",
                                                             "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    if DRY:
        log(f"DRY: would upload state from {src}"); return True
    rc, _ = kaggle("datasets", "status", f"{user}/{SSLUG}")
    if rc == 0:
        rc, out = kaggle("datasets", "version", "-p", str(stage), "-m", f"state {time.ctime()}", "--dir-mode", "zip")
    else:
        rc, out = kaggle("datasets", "create", "-p", str(stage), "--dir-mode", "zip")
    if rc != 0:
        log("STATE UPLOAD FAILED: " + out.strip()[-300:]); return False
    for _ in range(40):  # wait until the new version is ready before the kernel mounts it
        time.sleep(15)
        _, st = kaggle("datasets", "status", f"{user}/{SSLUG}")
        if "ready" in st.lower():
            log("state dataset ready"); return True
    log("state dataset not ready after 10 min; will retry next tick"); return False


def main():
    if (D / "DONE").exists() or (D / "STOP_NEEDS_DEBUG").exists():
        return
    st = json.loads(STATE_F.read_text()) if STATE_F.exists() else {"session": 0, "downloaded": 0, "fails": 0, "pushed_with_state": False}
    user = K.username()
    rc, out = kaggle("kernels", "status", f"{user}/{KSLUG}")
    s = out.lower()
    if st["session"] == 0 or (rc != 0 and "not found" in s):
        if push_kernel(user, with_state=False):
            st["session"] = 1
        STATE_F.write_text(json.dumps(st)); return
    if any(w in s for w in ("running", "queued")):
        log(f"session {st['session']} running"); status_md(f"Session {st['session']} is running on Kaggle.")
        return
    finished = any(w in s for w in ("complete", "error", "cancel"))
    if not finished:
        log("unknown status: " + out.strip()[-200:]); return

    n = st["session"]
    dst = D / f"sessions/{n:02d}"
    if st["downloaded"] < n:
        shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
        rc2, o2 = kaggle("kernels", "output", f"{user}/{KSLUG}", "-p", str(dst), "-o")
        if rc2 != 0:
            log("download failed: " + o2.strip()[-200:]); return
        st["downloaded"] = n
        failed = "error" in s or (dst / "error.txt").exists()
        st["fails"] = st["fails"] + 1 if failed else 0
        STATE_F.write_text(json.dumps(st))
        prog = (dst / "progress.log").read_text(encoding="utf-8") if (dst / "progress.log").exists() else ""
        log(f"session {n} finished ({'FAILED' if failed else 'ok'}); last progress: " + " | ".join(prog.strip().splitlines()[-3:]))
    prog = (dst / "progress.log").read_text(encoding="utf-8") if (dst / "progress.log").exists() else ""
    if "ALL DONE" in prog:
        shutil.copytree(dst, D / "final", dirs_exist_ok=True)
        (D / "DONE").write_text(time.ctime())
        done = re.findall(r"milestone \d+ done: .*", prog)
        status_md("ALL DONE.\n\n" + "\n".join(f"- {d}" for d in done))
        log("ALL DONE"); return
    if st["fails"] >= 2:
        (D / "STOP_NEEDS_DEBUG").write_text(f"two failed sessions in a row; see sessions/{n:02d}/error.txt")
        status_md(f"STOPPED: two failed sessions in a row. See `sessions/{n:02d}/error.txt`; fix and delete STOP_NEEDS_DEBUG.")
        log("STOP: two consecutive failures"); return
    # resume: newest session that produced state
    src = dst if (dst / "progress.log").exists() else None
    if src is None:
        prev = sorted(p for p in (D / "sessions").glob("*") if (p / "progress.log").exists())
        src = prev[-1] if prev else None
    if src is not None and not upload_state(user, src):
        return
    if push_kernel(user, with_state=src is not None):
        st["session"] = n + 1
        STATE_F.write_text(json.dumps(st))
        status_md(f"Session {n + 1} pushed (resuming from session {src.name if src else 'none'}).")


if __name__ == "__main__":
    main()
