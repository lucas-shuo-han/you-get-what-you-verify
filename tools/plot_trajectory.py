"""Plot one run's trajectory.jsonl and check it.

Canonical trajectory: lines are read in file order; a `resume` event with resume_from_step=R invalidates every
earlier train/validation/snapshot line with step > R (that branch was rolled back). Raw lines are never deleted.
Older lines without an `event` field (E1's first session) are classified from their keys.

Usage: python tools/plot_trajectory.py <run_dir> [--evals evals.json]  ->  <run_dir>/trajectory.png + check report
evals.json (optional): [{"step": 237, "validity": 0.8, "recognition": 0.0, "joint": 0.0}, ...]
"""
import argparse
import json
from pathlib import Path


def canonical(path):
    keep = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        e = json.loads(line)
        kind = e.get("event") or ("validation" if "eval_loss" in e else "train" if "loss" in e else "summary")
        e["event"] = kind
        if kind == "resume":
            r = e["resume_from_step"]
            for k in keep:
                if k["event"] in ("train", "validation", "snapshot") and k["step"] > r:
                    k["rolled_back"] = True
        keep.append(e)
    return [k for k in keep if not k.get("rolled_back")]


def check(ev, eval_every, milestones):
    tr = [e["step"] for e in ev if e["event"] == "train"]
    va = sorted({e["step"] for e in ev if e["event"] == "validation"})
    sn = {e["step"] for e in ev if e["event"] == "snapshot"}
    probs = []
    if len(tr) != len(set(tr)):
        probs.append("duplicate train steps after rollback handling")
    if 0 not in va:
        probs.append("no step-0 validation loss")
    for m in milestones:
        if m not in va:
            probs.append(f"milestone {m}: no validation loss at that step")
        if m not in sn:
            probs.append(f"milestone {m}: no snapshot at that step")
    if eval_every and va:
        gaps = [s for s in range(eval_every, max(va) + 1, eval_every) if s not in va]
        if gaps:
            probs.append(f"missing periodic validation at steps {gaps[:10]}{'...' if len(gaps) > 10 else ''}")
    ends = [e for e in ev if e["event"] == "end"]
    return {"train_points": len(tr), "validation_points": len(va), "snapshots": sorted(sn),
            "last_end": ends[-1] if ends else None, "problems": probs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--evals", default=None)
    ap.add_argument("--eval_every", type=int, default=40)
    ap.add_argument("--milestones", default="", help="comma-separated steps that must have val loss + snapshot")
    a = ap.parse_args()
    run = Path(a.run)
    ev = canonical(run / "trajectory.jsonl")
    ms = [int(x) for x in a.milestones.split(",") if x]
    rep = check(ev, a.eval_every, ms)
    (run / "trajectory_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evals = json.load(open(a.evals)) if a.evals else []
    n = 3 if evals else 1
    fig, ax = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True, squeeze=False)
    ax = ax[:, 0]
    tr = [e for e in ev if e["event"] == "train"]
    va = [e for e in ev if e["event"] == "validation"]
    ax[0].plot([e["step"] for e in tr], [e["loss"] for e in tr], lw=1, label="train loss (10-step mean)")
    ax[0].plot([e["step"] for e in va], [e["eval_loss"] for e in va], "o-", ms=3, label="validation loss (held-out)")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    if evals:
        st = [e["step"] for e in evals]
        ax[1].plot(st, [e.get("validity") for e in evals], "s-", label="physical validity")
        ax[1].set_ylabel("rate")
        ax[1].legend()
        ax[2].plot(st, [e.get("recognition") for e in evals], "^-", label="semantic success (unfiltered)")
        ax[2].plot(st, [e.get("joint") for e in evals], "v-", label="joint success")
        ax[2].set_ylabel("rate")
        ax[2].legend()
    ax[-1].set_xlabel("optimizer step")
    ax[0].set_title(run.name)
    fig.tight_layout()
    fig.savefig(run / "trajectory.png", dpi=130)
    print("wrote", run / "trajectory.png")


if __name__ == "__main__":
    main()
