"""Reproduce all paper figures from files in results/ and data/.  python paper/make_figs.py
One categorical palette (validated for CVD with the dataviz validator), fixed per group across every figure."""
import json, math, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageChops

R = Path(__file__).resolve().parents[1]; F = R / "paper/figs"; F.mkdir(exist_ok=True)
FULL, HALF = 5.5, 2.65
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL = {"base": "#9a9893", "A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "distill": "#eda100", "teacher": "#6250d6"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "savefig.bbox": None})

def wilson(k, n, z=1.96):
    if n == 0: return 0, 0, 0
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, p - max(0, c-h), min(1, c+h) - p

def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)

# 1 pipeline: generate -> verify -> (feedback | accept) -> SFT student -> evaluate
fig, ax = plt.subplots(figsize=(FULL, 1.25)); fig.subplots_adjust(0, 0, 1, 1); ax.axis("off"); ax.set_xlim(-0.05, 11.0); ax.set_ylim(0.7, 2.35)
W, H, Y = 1.95, 1.05, 1.25
boxes = [(0.0, "Caption", "text prompt", "base"), (2.25, "Generate", "large model\n(teacher)", "teacher"),
         (4.5, "Verify", "format, collision,\nfloating, Gurobi", None), (6.75, "SFT student", "LoRA on accepted\nstructures", "A"),
         (9.0, "Evaluate", "Gurobi validity,\nrecognition", None)]
for x, t, s, c in boxes:
    ec = COL[c] if c else INK2
    ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0,rounding_size=0.12", fc="white", ec=ec, lw=0.9))
    ax.text(x + W/2, Y + H - 0.27, t, ha="center", va="center", weight="bold", fontsize=8)
    ax.text(x + W/2, Y + 0.34, s, ha="center", va="center", fontsize=8, color=INK2, linespacing=1.1)
for i, x in enumerate((1.97, 4.22, 6.47, 8.72)):
    if i == 2: x0 = x
    ax.annotate("", (x + 0.27, Y + H/2), (x, Y + H/2), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=7))
ax.annotate("", (3.22, Y - 0.03), (5.47, Y - 0.03), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=COL["B"], mutation_scale=7,
            connectionstyle="arc3,rad=-0.25"))
ax.text(5.6, 0.82, "reject: checker error text fed back (up to 3 rounds)", ha="left", va="center", fontsize=8, color=INK2)
fig.savefig(F/"pipeline.pdf"); plt.close(fig)

# montage helpers: crop whitespace, pad to uniform square tiles
def _crop(im):
    bb = ImageChops.difference(im, Image.new("RGB", im.size, "white")).convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
    return im.crop(bb) if bb else im

def tile(path, size=(560, 300), stack=False):
    """Renders hold two views side by side: crop each view to content, join them tightly, fit into a uniform tile."""
    im = Image.open(path).convert("RGB"); w, h = im.size
    views = [_crop(im.crop((0, 0, w//2, h))), _crop(im.crop((w//2, 0, w, h)))]
    gap = 24
    if stack:
        vw = max(v.size[0] for v in views); vh = sum(v.size[1] for v in views) + gap
        joined = Image.new("RGB", (vw, vh), "white"); y = 0
        for v in views: joined.paste(v, ((vw - v.size[0])//2, y)); y += v.size[1] + gap
    else:
        vw = sum(v.size[0] for v in views) + gap; vh = max(v.size[1] for v in views)
        joined = Image.new("RGB", (vw, vh), "white"); x = 0
        for v in views: joined.paste(v, (x, vh - v.size[1])); x += v.size[0] + gap
    sc = min(size[0] * 0.97 / vw, size[1] * 0.97 / vh); joined = joined.resize((max(1, int(vw*sc)), max(1, int(vh*sc))), Image.LANCZOS)
    canvas = Image.new("RGB", size, "white"); canvas.paste(joined, ((size[0]-joined.size[0])//2, (size[1]-joined.size[1])//2)); return canvas

def montage(rows, fname, width, row_colors, ncol=None, size=(560, 300), stack=False, labw=0.0):
    """rows: (label or None, files). Rows longer than ncol wrap. labw: inches reserved left for horizontal row labels."""
    ncol = ncol or max(len(r[1]) for r in rows)
    grid = []
    for lab, files in rows:
        for k in range(0, len(files), ncol): grid.append((lab if k == 0 else "", files[k:k+ncol], lab))
    nrow = len(grid); tw = (width - labw) / ncol; th = tw * size[1] / size[0]; title_h = 0.17
    fig = plt.figure(figsize=(width, nrow * (th + title_h)))
    for i, (lab, files, glab) in enumerate(grid):
        for j in range(ncol):
            x0 = (labw + j * tw) / width; y0 = 1 - (i + 1) * (th + title_h) / (nrow * (th + title_h))
            a = fig.add_axes([x0 + 0.004, y0, tw / width - 0.008, th / (nrow * (th + title_h))])
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(False)
            if j >= len(files): a.axis("off"); continue
            a.imshow(tile(files[j], size, stack), aspect="auto")
            a.set_title(Path(files[j]).stem.split("_")[0] if glab else Path(files[j]).stem.rsplit("_", 1)[0], fontsize=8, pad=1.5, color=INK)
            if j == 0 and glab:
                a.spines["left"].set_visible(True); a.spines["left"].set_color(row_colors[[r[0] for r in rows].index(glab)]); a.spines["left"].set_linewidth(2.0)
                if lab: fig.text(0.0, y0 + th / (nrow * (th + title_h)) / 2, lab, ha="left", va="center", fontsize=8, color=INK)
    fig.savefig(F/fname, dpi=250); plt.close(fig)

# 2 montage: matched human vs distilled-SFT valid outputs
hs = sorted(glob.glob(str(R/"results/recognition_eval_sft_v2/renders/human/*.png")))[:5]
ss = sorted(glob.glob(str(R/"results/recognition_eval_sft_v2/renders/sft/*.png")))[:5]
montage([("Human data", hs), ("Distilled\nSFT (valid)", ss)], "solid_vs_human.pdf", FULL, [COL["C"], COL["distill"]], size=(300, 330), stack=True, labw=0.8)

# 3 recognition top-1 with Wilson 95% CI
rows = [("Human (random)", 15, 20, "base"), ("DeepSeek A", 14, 20, "teacher"), ("DeepSeek B", 14, 20, "teacher"),
        ("DeepSeek distill data", 12, 20, "teacher"), ("Opus", 9, 10, "teacher"), ("Human (matched)", 12, 14, "base"),
        ("Distilled SFT, valid", 0, 14, "distill")]
fig, ax = plt.subplots(figsize=(FULL * 0.72, 2.1)); fig.subplots_adjust(left=0.3, right=0.9, top=0.99, bottom=0.2)
for i, (lab, k, n, c) in enumerate(rows):
    p, lo, hi = wilson(k, n)
    ax.barh(i, p, color=COL[c], height=0.62, lw=0)
    ax.errorbar(p, i, xerr=[[lo], [hi]], color=INK, lw=0.7, capsize=1.8, capthick=0.7)
    ax.text(1.02, i, f"{k}/{n}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 1.0); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Blind top-1 recognition (21 classes)")
ax.axvline(1/21, ls=(0, (2, 2)), lw=0.8, color=INK2)
ax.text(1/21 + 0.012, -0.72, "chance 1/21", fontsize=8, color=INK2, va="center", ha="left")
ax.set_ylim(len(rows) - 0.5, -0.95)
fig.savefig(F/"recognition.pdf"); plt.close(fig)

# 4 physics-valid (Gurobi): left 90 prompts pooled seeds, right 28 new-category prompts per run
grp = [("Base", [0], "base"), ("A\n4k human", [15, 17], "A"), ("C\nA + human", [21, 18], "C"), ("B\nA + AI", [26, 20], "B")]
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.2), sharey=True, gridspec_kw={"width_ratios": [1, 1.15]}); fig.subplots_adjust(left=0.1, right=0.995, top=0.9, bottom=0.17, wspace=0.08)
ax = axs[0]
for i, (lab, ks, c) in enumerate(grp):
    K, N = sum(ks), 90 * len(ks); p, lo, hi = wilson(K, N)
    ax.bar(i, p, color=COL[c], width=0.62, lw=0)
    ax.errorbar(i, p, yerr=[[lo], [hi]], color=INK, lw=0.7, capsize=2, capthick=0.7)
    for j, k in enumerate(ks):
        ax.plot(i + (j - 0.5) * 0.2 * (len(ks) > 1), k / 90, "o", ms=3.2, mfc="white", mec=INK, mew=0.7, zorder=4)
    ax.text(i, p + hi + 0.012, f"{K}/{N}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(grp))); ax.set_xticklabels([g[0] for g in grp], fontsize=8)
ax.set_ylim(0, 0.42); fig.supylabel("Gurobi-valid at $k{=}0$", x=0.005, fontsize=8); ygrid(ax); ax.tick_params(axis="x", length=0)
ax.set_title("90 original-category prompts", fontsize=8, loc="left")
ax.plot([], [], "o", ms=3.2, mfc="white", mec=INK, mew=0.7, label="single seed"); ax.legend(frameon=False, fontsize=8, loc="upper left", handletextpad=0.2, borderaxespad=0.1)
gnc = [("Base", 0, 28, "base"), ("A s0", 3, 28, "A"), ("A s1", 5, 28, "A"), ("C s0", 4, 28, "C"), ("B s0", 1, 28, "B"), ("B s1", 4, 28, "B")]
ax = axs[1]
for i, (lab, k, n, c) in enumerate(gnc):
    p, lo, hi = wilson(k, n)
    ax.bar(i, p, color=COL[c], width=0.62, lw=0)
    ax.errorbar(i, p, yerr=[[lo], [hi]], color=INK, lw=0.7, capsize=2, capthick=0.7)
    ax.text(i, p + hi + 0.012, f"{k}/{n}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(gnc))); ax.set_xticklabels([x[0] for x in gnc], fontsize=8); ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 0.42); ygrid(ax); ax.set_title("28 new-category prompts (per run)", fontsize=8, loc="left")
fig.savefig(F/"physics_valid.pdf"); plt.close(fig)

# 5 training curves: smoothed over faint raw; end-of-training validation loss markers; direct labels
runs = [("A s0", "results/mix_A/sft_A/train_log.json", "A", "A_s0"), ("B s0", "results/mix_B/sft_B/train_log.json", "B", "B_s0"),
        ("B s1", "results/final_small/b1x1/sft_B_s1_x1/train_log.json", "B", "B_s1"),
        ("C s0", "results/final_small/c0x1/sft_C_s0_x1/train_log.json", "C", "C_s0")]
VAL = json.load(open(R/"results/paper/as1/val_loss_4adapters.json"))["adapters"]
def ema(y, w=5):
    y = np.asarray(y); h = w // 2
    return [y[max(0, i-h):i+h+1].mean() for i in range(len(y))]
fig, ax = plt.subplots(figsize=(FULL, 2.45)); fig.subplots_adjust(left=0.1, right=0.8, top=0.97, bottom=0.33)
ends = []
for lab, f, c, vk in runs:
    L = json.load(open(R/f))["log_history"]; st = [h["step"] for h in L if "loss" in h]; lo = [h["loss"] for h in L if "loss" in h]
    ls = "--" if lab == "B s1" else "-"
    ax.plot(st, lo, color=COL[c], lw=0.5, alpha=0.28)
    ax.plot(st, ema(lo), color=COL[c], lw=1.4, ls=ls, label=lab)
    v = VAL[vk]["loss"]; ax.plot(st[-1] + 10, v, "D", ms=4, color=COL[c], mec="white", mew=0.6, zorder=5, clip_on=False)
    ends.append((lab, v, c)); print(lab, "first", round(lo[0], 3), "last", round(lo[-1], 3), "val", round(v, 4))
ax.set_xlim(0, st[-1] + 2); ax.set_xlabel("optimizer step (effective batch 16)"); ax.set_ylabel("loss (token cross-entropy)")
ygrid(ax)
# held-out validation values as a small table at the right, sorted, one line each (no stacked callouts)
tx = 1.06; ax.text(tx, 0.98, "held-out val. loss", transform=ax.transAxes, fontsize=8, color=INK2, va="top")
for k, (lab, v, c) in enumerate(sorted(ends, key=lambda e: e[1])):
    ax.text(tx, 0.86 - k * 0.11, f"{lab}  {v:.3f}", transform=ax.transAxes, fontsize=8, color=COL[c], va="top")
ax.text(tx, 0.86 - 4 * 0.11, f"base  {0.465:.3f}", transform=ax.transAxes, fontsize=8, color=INK2, va="top")
h, l = ax.get_legend_handles_labels()
h += [plt.Line2D([], [], color=INK2, lw=0.5, alpha=0.4), plt.Line2D([], [], marker="D", ms=4, color=INK2, ls="none")]
l += ["raw train loss", "val. loss (end)"]
fig.legend(h, l, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.6, handletextpad=0.4)
fig.savefig(F/"train_curves.pdf"); plt.close(fig)

# 6 gallery of Opus-harness structures (new categories)
pick = sorted(glob.glob(str(R/"data/opus_gen/newcat_0/renders/*.png")))[:3] + sorted(glob.glob(str(R/"data/opus_gen/newcat_1/renders/*.png")))[:3]
montage([(None, pick)], "opus_gallery.pdf", FULL, [COL["teacher"]], ncol=3, size=(560, 270))
print("ok", sorted(p.name for p in F.iterdir()))
