"""§4k night rounds: summaries (r2, r3, r4, r5) + figures. Reads only saved results/scores (bit-reproducible).

python -m brickexp.truss.night r2|r3|r4|r5|figs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..screen.conformal import threshold, metrics, id_tolerance, auc, _rank_avg
from .gen import FAMILIES10, THREE_D

import os
D = Path(os.environ.get("TRUSS_OUT", "results/truss10"))
FIG = Path("results/figs"); FIG.mkdir(parents=True, exist_ok=True)
A5 = 0.05
SEEDS = (0, 1, 2)


def npz(tag, s):
    return np.load(D / f"{tag}_s{s}_scores.npz", allow_pickle=True)


def pass_rate(sc, y, tau):
    y0 = np.asarray(y) == 0
    return float((np.asarray(sc)[y0] < tau).mean()) if y0.any() else float("nan")


def n0(tag, alpha=A5):
    """per-seed (miss, pass, saved, auc) at N=0 from saved scores."""
    out = []
    for s in SEEDS:
        z = npz(tag, s)
        tau = threshold(z["ca"], z["ya"], alpha); m = metrics(z["te"], z["yt"], tau)
        out.append({"miss": m["miss"], "pass": pass_rate(z["te"], z["yt"], tau), "saved": m["saved"], "auc": m["auc"],
                    "id_miss": metrics(z["va"], z["yv"], tau)["miss"], "id_auc": auc(z["va"], z["yv"])})
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]} | {
        k + "_range": [float(min(o[k] for o in out)), float(max(o[k] for o in out))] for k in ("miss", "pass")}


def verdict(ok, n):
    return "TRANSFER" if ok >= .75 * n else ("NO_TRANSFER" if ok < .5 * n else "PARTIAL")


def r2():
    T = {f: n0(f"loco_{f}") for f in FAMILIES10}
    T["size"] = n0("size")
    for f in FAMILIES10:
        T[f]["signal_geo"] = json.loads((D / f"loco_{f}.json").read_text())["signal"]["outlier_rate"]
    ok = {f: T[f]["miss"] <= .10 and T[f]["pass"] >= .5 for f in FAMILIES10}
    V = {"n_ok": sum(ok.values()), "n": 10, "verdict": verdict(sum(ok.values()), 10), "ok": ok,
         "three_d": {f: {k: T[f][k] for k in ("miss", "pass", "auc")} for f in THREE_D},
         "size_ok": T["size"]["miss"] <= .10 and T["size"]["pass"] >= .5,
         "positives_miss_ge_10": [f for f in FAMILIES10 if T[f]["miss"] >= .10]}
    pos = np.array([f in V["positives_miss_ge_10"] for f in FAMILIES10])
    g = [T[f]["signal_geo"] for f in FAMILIES10]
    V["geo_signal_auroc"] = auc(np.array(g), pos) if 0 < pos.sum() < 10 else None
    (D / "r2_summary.json").write_text(json.dumps({"verdict": V, "table": T}, indent=1))
    print(json.dumps(V, indent=1))
    for f, t in T.items():
        print(f"{f:10s} miss {t['miss']:.3f} ({t['miss_range'][0]:.3f}-{t['miss_range'][1]:.3f}) pass {t['pass']:.3f} "
              f"saved {t['saved']:.3f} auc {t['auc']:.3f} id_miss {t['id_miss']:.3f}")


def r3():
    S = json.loads((D / "r2_summary.json").read_text())
    targets = [f for f, o in S["verdict"]["ok"].items() if not o] + ["size"]
    NS = [0, 50, 100, 300, 1000]; R = {}
    for t in targets:
        tag = "size" if t == "size" else f"loco_{t}"
        R[t] = {}
        for alpha in (0.01, 0.05):
            curve = {}
            for N in NS:
                rows = []
                for s in SEEDS:
                    z = npz(tag, s); rng = np.random.default_rng(7000 + s)
                    if N == 0:
                        tau = threshold(z["ca"], z["ya"], alpha); m = metrics(z["te"], z["yt"], tau)
                        rows.append({"miss": m["miss"], "pass": pass_rate(z["te"], z["yt"], tau), "infeasible": m["infeasible"],
                                     "tol": id_tolerance(alpha, int(z["ya"].sum()), int(z["yt"].sum()))})
                        continue
                    if N > len(z["po"]):
                        continue
                    for _ in range(50):
                        j = rng.choice(len(z["po"]), N, replace=False)
                        tau = threshold(z["po"][j], z["yp"][j], alpha); m = metrics(z["te"], z["yt"], tau)
                        rows.append({"miss": m["miss"], "pass": pass_rate(z["te"], z["yt"], tau), "infeasible": m["infeasible"],
                                     "tol": id_tolerance(alpha, int(z["yp"][j].sum()), int(z["yt"].sum()))})
                if rows:
                    c = {k: float(np.mean([r[k] for r in rows])) for k in ("miss", "pass", "infeasible", "tol")}
                    c["miss_p95"] = float(np.percentile([r["miss"] for r in rows], 95))
                    c["ok"] = c["miss"] <= alpha + c["tol"] and c["pass"] >= .5
                    curve[N] = c
            nstar = next((N for N in NS if N in curve and curve[N]["ok"]), None)
            R[t][f"alpha_{alpha}"] = {"curve": curve, "N_star": nstar}
    fam_targets = [t for t in targets if t != "size"]
    ns = [R[t]["alpha_0.05"]["N_star"] for t in fam_targets]
    finite = [n for n in ns if n is not None]
    med = float(np.median([n if n is not None else np.inf for n in ns])) if ns else None
    if not ns:
        v = "NO_FAILING_FAMILIES"
    elif len(finite) < len(ns) / 2 or med > 1000:
        v = "COSTLY"
    elif med <= 300:
        v = "DEPLOYABLE"
    else:
        v = "BETWEEN(300,1000]"
    out = {"verdict": v, "median_N_star_alpha5": med, "N_star": {t: R[t]["alpha_0.05"]["N_star"] for t in targets},
           "N_star_alpha1": {t: R[t]["alpha_0.01"]["N_star"] for t in targets}, "curves": R}
    (D / "r3_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("verdict", "median_N_star_alpha5", "N_star", "N_star_alpha1")}, indent=1))


def r4():
    S = json.loads((D / "r2_summary.json").read_text())
    pos = S["verdict"]["positives_miss_ge_10"]
    rows = {}
    for f in FAMILIES10:
        J = json.loads((D / f"loco_{f}_aux.json").read_text())
        sig = [J["seeds"][str(s)]["signal_mech"] for s in SEEDS]
        plain, aux = S["table"][f], n0(f"loco_{f}_aux")
        rows[f] = {"mech_primary": float(np.mean([x["primary"] for x in sig])), "ks": float(np.mean([x["ks_slender"] for x in sig])),
                   "aux_acc_target": float(np.mean([x["aux_acc_target"] for x in sig])),
                   "geo": plain["signal_geo"], "pos": f in pos,
                   "id_miss_plain": plain["id_miss"], "id_miss_aux": aux["id_miss"],
                   "id_auc_plain": plain["id_auc"], "id_auc_aux": aux["id_auc"], "miss_aux": aux["miss"], "miss_plain": plain["miss"]}
    y = np.array([rows[f]["pos"] for f in FAMILIES10])
    exc = [S["table"][f]["miss"] - A5 for f in FAMILIES10]
    sp = lambda x: float(np.corrcoef(_rank_avg(np.array(x, float)), _rank_avg(np.array(exc, float)))[0, 1])
    V = {"n_pos": int(y.sum()), "positives": pos}
    for k in ("mech_primary", "ks", "geo"):
        x = [rows[f][k] for f in FAMILIES10]
        V[f"auroc_{k}"] = auc(np.array(x), y) if 0 < y.sum() < 10 else None
        V[f"spearman_{k}"] = sp(x)
    V["verdict"] = (("EFFECTIVE" if V["auroc_mech_primary"] >= .8 else "NOT_EFFECTIVE") if y.sum() >= 3 else "FEW_POSITIVES(spearman)")
    d_miss = np.mean([rows[f]["id_miss_aux"] - rows[f]["id_miss_plain"] for f in FAMILIES10])
    d_auc = np.mean([rows[f]["id_auc_plain"] - rows[f]["id_auc_aux"] for f in FAMILIES10])
    V["id_check"] = {"mean_id_miss_increase": float(d_miss), "mean_id_auc_drop": float(d_auc),
                     "pass": bool(d_miss <= .01 and d_auc <= .01)}
    (D / "r4_summary.json").write_text(json.dumps({"verdict": V, "rows": rows}, indent=1))
    print(json.dumps(V, indent=1))
    for f, r in rows.items():
        print(f"{f:10s} pos {int(r['pos'])} mech {r['mech_primary']:.3f} ks {r['ks']:.3f} geo {r['geo']:.3f} acc {r['aux_acc_target']:.3f} "
              f"miss plain/aux {r['miss_plain']:.3f}/{r['miss_aux']:.3f}")


def r5():
    res = {}
    for arm in ("aug_flip", "aug_plain"):
        tag = f"loco_howe_{arm}"; per = []
        for s in SEEDS:
            z = npz(tag, s); tau = threshold(z["ca"], z["ya"], A5)
            fam_miss = {}
            for f in sorted(set(z["fv"].tolist())):
                k = z["fv"] == f
                fam_miss[f] = metrics(z["va"][k], z["yv"][k], tau)["miss"]
            per.append({"howe_miss": metrics(z["te"], z["yt"], tau)["miss"], "howe_pass": pass_rate(z["te"], z["yt"], tau),
                        "other_mean_miss": float(np.nanmean(list(fam_miss.values()))), "fam_miss": fam_miss})
        res[arm] = {"howe_miss": float(np.mean([p["howe_miss"] for p in per])),
                    "howe_miss_range": [float(min(p["howe_miss"] for p in per)), float(max(p["howe_miss"] for p in per))],
                    "howe_pass": float(np.mean([p["howe_pass"] for p in per])),
                    "other_mean_miss": float(np.mean([p["other_mean_miss"] for p in per])), "per_seed": per}
    rise = res["aug_flip"]["other_mean_miss"] - res["aug_plain"]["other_mean_miss"]
    hm = res["aug_flip"]["howe_miss"]
    v = "SUPPORTS_SHORTCUT" if hm <= .15 and rise <= .02 else ("REJECTS" if hm > .25 else "UNCERTAIN")
    out = {"verdict": v, "howe_miss_flip": hm, "howe_miss_control": res["aug_plain"]["howe_miss"],
           "other_mean_miss_rise": rise, "note": "mechanism test, not a deployment method (flips can create Howe-like panels)",
           "arms": res}
    (D / "r5_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("verdict", "howe_miss_flip", "howe_miss_control", "other_mean_miss_rise")}, indent=1))


def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    C1, C2, GR = "#3b6ea8", "#d9822b", "#888888"
    # (a) LEGO leave-one-category-out
    L = json.loads(Path("results/screen_loco/summary.json").read_text())["table"]
    L = sorted(L, key=lambda r: -r["miss"])
    g = json.loads(Path("results/screen/results.json").read_text())["summary"]["alpha_0.05"]["cat_03467517"]["N0"]["miss"]
    names = ["guitar (dev)"] + [r["cat"] for r in L]; vals = [g] + [r["miss"] for r in L]
    lo = [0] + [r["miss"] - r["miss_range"][0] for r in L]; hi = [0] + [r["miss_range"][1] - r["miss"] for r in L]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(range(len(vals)), vals, color=[C2] + [C1] * len(L), yerr=[lo, hi], capsize=2, ecolor=GR)
    ax.axhline(.05, color="k", lw=1, ls="--", label="α = 5%"); ax.axhline(.10, color="r", lw=1, ls=":", label="2α")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("miss rate (α=5%, N=0)"); ax.set_title("LEGO screener: leave-one-category-out (3 seeds, range bars)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "a_lego_loco_miss.png", dpi=200); plt.close(fig)
    # (b) truss round 2 LOCO
    if not (D / "r2_summary.json").exists():
        return print("r2 not ready; only (a) drawn")
    T = json.loads((D / "r2_summary.json").read_text())["table"]
    ks = FAMILIES10 + ["size"]
    fig, ax = plt.subplots(figsize=(8, 3.2)); x = np.arange(len(ks)); w = .38
    m = [T[k]["miss"] for k in ks]; p = [T[k]["pass"] for k in ks]
    em = [[T[k]["miss"] - T[k]["miss_range"][0] for k in ks], [T[k]["miss_range"][1] - T[k]["miss"] for k in ks]]
    ax.bar(x - w / 2, m, w, color=C2, label="miss rate", yerr=em, capsize=2, ecolor=GR)
    ax.bar(x + w / 2, p, w, color=C1, label="safe pass rate")
    ax.axhline(.10, color="r", lw=1, ls=":", label="miss ≤ 10% bar"); ax.axhline(.5, color="k", lw=1, ls="--", label="pass ≥ 50% bar")
    ax.set_xticks(x); ax.set_xticklabels([k if k != "size" else "size shift" for k in ks], rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Truss screener: leave-one-family-out, 10 families (α=5%, N=0, 3 seeds)")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center"); fig.tight_layout(); fig.savefig(FIG / "b_truss_loco.png", dpi=200); plt.close(fig)
    # (c) label efficiency
    p3 = D / "r3_summary.json"
    if p3.exists():
        R = json.loads(p3.read_text())["curves"]
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        for t, c in R.items():
            cv = c["alpha_0.05"]["curve"]; Ns = sorted(int(n) for n in cv)
            xs = [max(n, 5) for n in Ns]
            axs[0].plot(xs, [cv[str(n)]["miss"] for n in Ns], "o-", label=t)
            axs[1].plot(xs, [cv[str(n)]["pass"] for n in Ns], "o-", label=t)
        for a_, yl, ln in ((axs[0], "miss rate", .05), (axs[1], "safe pass rate", .5)):
            a_.set_xscale("log"); a_.axhline(ln, color="k", ls="--", lw=1); a_.set_xlabel("target labels N (N=0 plotted at 5)"); a_.set_ylabel(yl)
        axs[0].legend(frameon=False, fontsize=7); fig.suptitle("Label efficiency of recalibration (α=5%, 50 draws × 3 seeds)")
        fig.tight_layout(); fig.savefig(FIG / "c_label_efficiency.png", dpi=200); plt.close(fig)
    # (d) geometric vs mechanism warning
    p4 = D / "r4_summary.json"
    if p4.exists():
        R4 = json.loads(p4.read_text()); rows = R4["rows"]; V = R4["verdict"]
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        for a_, k, nm in ((axs[0], "geo", "geometric outlier rate"), (axs[1], "mech_primary", "mechanism signal |Δ comp-web frac|")):
            for f, r in rows.items():
                a_.scatter(r[k], T[f]["miss"], color=C2 if r["pos"] else C1); a_.annotate(f, (r[k], T[f]["miss"]), fontsize=7)
            au = V.get(f"auroc_{k}")
            a_.set_xlabel(nm); a_.set_ylabel("miss rate (α=5%)"); a_.axhline(.10, color="r", ls=":", lw=1)
            a_.set_title(f"AUROC = {au:.2f}" if au is not None else f"ρ = {V[f'spearman_{k}']:.2f}")
        fig.suptitle("Label-free early warning: geometric vs mechanism-aware"); fig.tight_layout()
        fig.savefig(FIG / "d_warning_signals.png", dpi=200); plt.close(fig)
    print(sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    {"r2": r2, "r3": r3, "r4": r4, "r5": r5, "figs": figs}[sys.argv[1]]()
