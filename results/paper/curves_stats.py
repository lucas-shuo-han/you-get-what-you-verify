"""(1) Training-loss curves of all mix runs on one figure. (2) Pass rates (local Gurobi, k=0) with Wilson 95% CIs and
Fisher exact tests. Writes results/paper/train_curves.png, results/paper/stats_table.md, results/paper/stats.json."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]; P = ROOT / "results/paper"
LOGS = {"A_s0": "mix_A/sft_A", "B_s0": "mix_B/sft_B", "B_s1": "final_small/b1x1/sft_B_s1_x1", "C_s0": "final_small/c0x1/sft_C_s0_x1"}
ex = P / "extra_logs.json"
if ex.exists(): LOGS.update(json.load(open(ex)))
fig, ax = plt.subplots(figsize=(6, 3.6))
for name, d in LOGS.items():
    L = json.load(open(ROOT / "results" / d / "train_log.json"))["log_history"]
    xs = [h["step"] for h in L if "loss" in h]; ys = [h["loss"] for h in L if "loss" in h]
    ax.plot(xs, ys, label=f"{name} (final {ys[-1]:.3f})", lw=1.3)
ax.set_xlabel("optimizer step (bs 16)"); ax.set_ylabel("training loss"); ax.set_yscale("log")
ax.set_title("LoRA SFT of Qwen2.5-1.5B: A = 4000 human; B = A + 3x AI; C = A + 3x extra human", fontsize=8)
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(P / "train_curves.png", dpi=200)

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
G = json.load(open(P / "gurobi_all.json"))
rows, S = [], {}
for key, v in G.items():
    lo, hi = wilson(v["gurobi_ok"], v["n"]); S[key] = {"k": v["gurobi_ok"], "n": v["n"], "wilson95": [lo, hi], "conn": v["connectivity_ok"]}
    rows.append(f"| {key} | {v['gurobi_ok']}/{v['n']} ({v['gurobi_ok']/v['n']:.0%}) | [{lo:.0%}, {hi:.0%}] | {v['connectivity_ok']}/{v['n']} |")
def fisher(a, b, alt="two-sided"):
    ka, na, kb, nb = S[a]["k"], S[a]["n"], S[b]["k"], S[b]["n"]
    return fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative=alt)[1]
pairs = [(a, b) for a, b in [("A_s0_90", "base_90"), ("B_s0_90", "A_s0_90"), ("B_s1_90", "A_s0_90"), ("C_s0_90", "A_s0_90"),
                             ("B_s0_90", "C_s0_90"), ("B_s1_90", "C_s0_90"), ("B_s0_90", "B_s1_90"), ("A_s1_90", "A_s0_90"),
                             ("B_s0_newcat", "A_s0_newcat")] if a in S and b in S]
F = {f"{a} vs {b}": {"two_sided_p": fisher(a, b), "one_sided_p(a>b)": fisher(a, b, "greater")} for a, b in pairs}
# seeds pooled where available (§4m style): B seeds vs C / A
md = ["| run (k=0) | Gurobi pass | Wilson 95% CI | connectivity pass |", "|---|---|---|---|"] + rows
md += ["", "| comparison | Fisher two-sided p | one-sided p (first > second) |", "|---|---|---|"]
md += [f"| {k} | {v['two_sided_p']:.3g} | {v['one_sided_p(a>b)']:.3g} |" for k, v in F.items()]
(P / "stats_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
json.dump({"rates": S, "fisher": F}, open(P / "stats.json", "w"), indent=1)
print("\n".join(md))
