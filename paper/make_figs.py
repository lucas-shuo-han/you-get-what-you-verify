"""Reproduce all paper figures from files in results/ and data/.  python paper/make_figs.py"""
import json, math, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
R = Path(__file__).resolve().parents[1]; F = R / "paper/figs"; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = {"human": "#5F5E5A", "ai": "#534AB7", "small": "#1D9E75", "base": "#B4B2A9", "warn": "#D85A30"}

def wilson(k, n, z=1.96):
    if n == 0: return 0, 0, 0
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, p - max(0, c-h), min(1, c+h) - p

# 1 pipeline
fig, ax = plt.subplots(figsize=(6.8, 1.9)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 2.4)
boxes = [(0.1, "Caption", "text prompt"), (2.1, "Generate", "large model"), (4.1, "Verify", "format, collision,\nfloating, Gurobi"),
         (6.1, "Check", "blind recognition"), (8.1, "Filter + SFT", "LoRA, small model")]
for x, t, s in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.7), 1.7, 1.1, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#EEEDFE", ec="#534AB7", lw=0.8))
    ax.text(x+0.85, 1.45, t, ha="center", va="center", weight="bold"); ax.text(x+0.85, 1.0, s, ha="center", va="center", fontsize=7, color="#3C3489")
for x in (1.8, 3.8, 5.8, 7.8):
    ax.annotate("", (x+0.3, 1.25), (x, 1.25), arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("", (3.0, 0.65), (5.0, 0.65), arrowprops=dict(arrowstyle="->", lw=0.8, color=C["warn"], connectionstyle="arc3,rad=-0.35"))
ax.text(4.0, 0.12, "checker error message fed back (<=3 rounds)", ha="center", fontsize=7, color=C["warn"])
fig.savefig(F/"pipeline.pdf", bbox_inches="tight"); plt.close(fig)

# 2 montage: distilled-SFT valid outputs vs matched human
def montage(dirpath, n=5):
    fs = sorted(glob.glob(str(dirpath/"*.png")))[:n]
    return [Image.open(f).convert("RGB") for f in fs], [Path(f).stem.split("_")[0] for f in fs]
hs, hl = montage(R/"results/recognition_eval_sft_v2/renders/human"); ss, sl = montage(R/"results/recognition_eval_sft_v2/renders/sft")
fig, axs = plt.subplots(2, 5, figsize=(7, 2.3))
for j in range(5):
    for i, (ims, labs, row) in enumerate(((hs, hl, "Human data"), (ss, sl, "Distilled SFT (valid)"))):
        a = axs[i, j]; a.imshow(ims[j]); a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
        a.set_title(labs[j], fontsize=7, pad=1)
        if j == 0: a.set_ylabel(row, fontsize=7)
fig.savefig(F/"solid_vs_human.pdf", bbox_inches="tight", dpi=200); plt.close(fig)

# 3 recognition top-1 with Wilson 95% CI
rows = [("Human (random)", 15, 20, "human"), ("DeepSeek A", 14, 20, "ai"), ("DeepSeek B", 14, 20, "ai"), ("DeepSeek distill data", 12, 20, "ai"),
        ("Opus", 9, 10, "ai"), ("Human (matched)", 12, 14, "human"), ("Distilled SFT, valid", 0, 14, "small")]
fig, ax = plt.subplots(figsize=(4.2, 2.4))
for i, (n_, k, n, c) in enumerate(rows):
    p, lo, hi = wilson(k, n); ax.barh(i, p, color=C[c], height=0.6); ax.errorbar(p, i, xerr=[[lo], [hi]], color="k", lw=0.7, capsize=2)
    ax.text(1.02, i, f"{k}/{n}", va="center", fontsize=7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 1.12)
ax.set_xlabel("Blind top-1 recognition (21 classes), 95% Wilson CI"); ax.axvline(1/21, ls=":", lw=0.7, color="k")
fig.savefig(F/"recognition.pdf", bbox_inches="tight"); plt.close(fig)

# 4 physics-valid (Gurobi) on 90 original prompts and 28 new-category prompts (k=0)
NL = chr(10)
g90 = [("Base (untrained)", 0, 90, "base"), ("Distilled SFT" + NL + "(96 DeepSeek)", 12, 90, "small"), ("A: 4k human", 15, 90, "human"),
       ("C: A+284 human" + NL + "x3 (seed 0)", 21, 90, "human"), ("B: A+284 AI x3" + NL + "(seed 0)", 26, 90, "ai"), ("B (seed 1)", 20, 90, "ai")]
gnc = [("Base", 0, 28, "base"), ("A", 3, 28, "human"), ("C s0", 4, 28, "human"), ("B s0", 1, 28, "ai"), ("B s1", 4, 28, "ai")]
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.3), gridspec_kw={"width_ratios": [6, 4]})
for ax, g, ttl in ((axs[0], g90, "90 original-category prompts"), (axs[1], gnc, "28 new-category prompts")):
    for i, (n_, k, n, c) in enumerate(g):
        p, lo, hi = wilson(k, n); ax.bar(i, p, color=C[c], width=0.65); ax.errorbar(i, p, yerr=[[lo], [hi]], color="k", lw=0.7, capsize=2)
        ax.text(i, p + hi + 0.015, f"{k}/{n}", ha="center", fontsize=7)
    ax.set_xticks(range(len(g))); ax.set_xticklabels([x[0] for x in g], rotation=25, ha="right", fontsize=6.5); ax.set_title(ttl, fontsize=8); ax.set_ylim(0, 0.5)
axs[0].set_ylabel("Gurobi-valid at k=0 (95% Wilson CI)")
fig.savefig(F/"physics_valid.pdf", bbox_inches="tight"); (R/"docs/img").mkdir(parents=True, exist_ok=True); fig.savefig(R/"docs/img/physics_valid.png", bbox_inches="tight", dpi=160); plt.close(fig)

# 5 training curves: A, B s0, B s1, C s0
runs = [("A (seed 0)", "results/mix_A/sft_A/train_log.json", "human", "-"), ("B (seed 0)", "results/mix_B/sft_B/train_log.json", "ai", "-"),
        ("B (seed 1)", "results/final_small/b1x1/sft_B_s1_x1/train_log.json", "ai", "--"), ("C (seed 0)", "results/final_small/c0x1/sft_C_s0_x1/train_log.json", "human", "--")]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for lab, f, c, ls in runs:
    L = json.load(open(R/f))["log_history"]; st = [h["step"] for h in L if "loss" in h]; lo = [h["loss"] for h in L if "loss" in h]
    ax.plot(st, lo, color=C[c], lw=1.1, ls=ls, label=lab); print(lab, "first", round(lo[0],3), "last", round(lo[-1],3))
ax.set_xlabel("optimizer step"); ax.set_ylabel("train loss (token CE)"); ax.legend(fontsize=6.5, frameon=False)
fig.savefig(F/"train_curve.pdf", bbox_inches="tight"); plt.close(fig)

# 6 gallery of Opus-harness structures (new categories + dataset categories)
pick = sorted(glob.glob(str(R/"data/opus_gen/newcat_0/renders/*.png")))[:3] + sorted(glob.glob(str(R/"data/opus_gen/newcat_1/renders/*.png")))[:3]
fig, axs = plt.subplots(1, len(pick), figsize=(7, 1.4))
for a, f in zip(axs, pick):
    a.imshow(Image.open(f).convert("RGB")); a.axis("off"); a.set_title(Path(f).stem.rsplit("_", 1)[0], fontsize=7, pad=1)
fig.savefig(F/"opus_gallery.pdf", bbox_inches="tight", dpi=200); plt.close(fig)
print("ok", sorted(p.name for p in F.iterdir()))
