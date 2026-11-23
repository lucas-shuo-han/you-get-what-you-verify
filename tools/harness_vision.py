"""Vision-feedback harness test: A (physics error text only) vs B (physics error + 2-view render + "does it look like X?").

Model deepseek-flash (DeepSeek-V4.1-Flash, image input), reasoning_effort=low, max_tokens 16000. 20 train-split prompts
(disjoint from the 90 eval prompts), same for both arms; <= 3 feedback rounds (4 attempts).
Stop: A when physics passes; B when physics passes AND the model replies DONE (judges it looks like the description).
Final structure = last physics-valid attempt (else the last attempt).
Key read inside this process only; never printed/logged/written. Spend guard via account balance.

python tools/harness_vision.py --n 20 --max_spend 4 --out results/harness_vision
"""
import argparse
import base64
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from brickexp import prompts  # noqa: E402
from brickexp.checker import check, extract_bricks_text, parse  # noqa: E402
from distill_filter import feats, keep  # noqa: E402
from distill_iou import vox  # noqa: E402
from distill_shape import aligned_iou  # noqa: E402

API = "https://api.deepseek.com"
LOCK = threading.Lock()  # matplotlib + Gurobi env creation are not thread-safe enough; serialize them


def _key():
    return (Path.home() / ".deepseek_key").read_text().strip()


def balance():
    r = requests.get(f"{API}/user/balance", headers={"Authorization": f"Bearer {_key()}"}, timeout=30)
    for b in r.json().get("balance_infos", []):
        if b.get("currency") == "CNY":
            return float(b["total_balance"])


def render_png(text, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    b = parse(text)[0]
    v = np.zeros((20, 20, 20), bool); c = np.empty((20, 20, 20, 4)); cm = plt.get_cmap("tab20")
    for k, x in enumerate(b):
        v[x.x:x.x + x.h, x.y:x.y + x.w, x.z] = True; c[x.x:x.x + x.h, x.y:x.y + x.w, x.z] = cm(k % 20)
    fig = plt.figure(figsize=(6, 3))
    for i, (el, az) in enumerate(((25, -60), (25, 120))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.voxels(v, facecolors=c, edgecolor=(0, 0, 0, .25), linewidth=.3)
        ax.set_xlim(0, 20); ax.set_ylim(0, 20); ax.set_zlim(0, 20)  # fixed grid, never autoscale to the model
        ax.set_box_aspect((1, 1, 1)); ax.view_init(el, az); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return base64.b64encode(Path(path).read_bytes()).decode()


class Client:
    def __init__(self, max_tokens, max_spend, log):
        self.max_tokens, self.max_spend, self.log = max_tokens, max_spend, log
        self.b0 = balance(); self.calls = []
        log({"event": "start", "balance_cny": self.b0})

    def spent(self):
        return round(self.b0 - balance(), 4)

    def chat(self, messages, tag):
        if len(self.calls) % 10 == 0 and self.spent() >= self.max_spend:
            raise RuntimeError("spend guard")
        t = time.time()
        for tries in range(3):
            try:
                r = requests.post(f"{API}/chat/completions", timeout=600,
                                  headers={"Authorization": f"Bearer {_key()}"},
                                  json={"model": "deepseek-flash", "messages": messages, "max_tokens": self.max_tokens,
                                        "reasoning_effort": "low", "temperature": 0.6})
                r.raise_for_status(); break
            except requests.RequestException as e:
                self.log({"event": "http_error", "status": getattr(e.response, "status_code", None), "tag": tag}); time.sleep(5)
        else:
            return ""
        j = r.json(); u = j.get("usage", {})
        rec = {"tag": tag, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
               "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
               "cache_hit": u.get("prompt_cache_hit_tokens"), "finish": j["choices"][0].get("finish_reason"),
               "sec": round(time.time() - t, 1)}
        self.calls.append(rec); self.log({"event": "call", **rec})
        return j["choices"][0]["message"].get("content") or ""


def episode(cl, rec, arm, out, max_fb=3):
    d = out / "renders" / arm / rec["structure_id"][:8]; d.mkdir(parents=True, exist_ok=True)
    base = prompts.initial_messages(rec["caption"])
    if arm == "B":
        base[-1]["content"] += ("\nAfter each attempt you will be shown renders of your structure and the physics "
                                "checker result. If it is physically valid and looks like the description, reply "
                                "with only the word DONE.")
    last_asst, last_fb, att, final = None, None, [], None
    for k in range(max_fb + 1):
        msgs = list(base)
        if last_asst is not None:  # window 1: initial prompt + last attempt + its feedback (both arms)
            msgs += [{"role": "assistant", "content": last_asst}, last_fb]
        raw = cl.chat(msgs, f"{arm}:{rec['structure_id'][:8]}:{k}")
        text = extract_bricks_text(raw)
        if arm == "B" and k > 0 and not text and "DONE" in raw.upper() and final is not None:
            att.append({"k": k, "done": True}); break
        with LOCK:
            r = check(raw, use_gurobi=True).to_dict()
            img = render_png(text, d / f"round{k}.png") if text and parse(text)[0] else None
        a = {"k": k, "ok": r["ok"], "failure": r["failure"], "message": r["message"], "text": text}
        att.append(a)
        if r["ok"]:
            final = a
            if arm == "A":
                break
        phys = "The physics checker found no problems." if r["ok"] else f"Physics problem: {r['message']}"
        if arm == "A":
            last_fb = {"role": "user", "content": prompts.feedback(r["message"])}
        else:
            txt = (f"This is the structure you built for \"{rec['caption']}\" (two views). Does it look like it? "
                   f"What does not look right? {phys} If it is physically valid and looks like the description, "
                   "reply only DONE; otherwise output the complete corrected structure (all bricks, one per line).")
            content = [{"type": "text", "text": txt}]
            if img:
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}})
            last_fb = {"role": "user", "content": content}
        last_asst = raw
    final = final or (att[-1] if att and "text" in att[-1] else None)
    res = {"arm": arm, "structure_id": rec["structure_id"], "bucket": rec["bucket"], "caption": rec["caption"],
           "attempts": att, "pass": bool(final and final.get("ok")), "rounds": len(att)}
    if final and final.get("text") and parse(final["text"])[0]:
        f = feats(final["text"]); res.update(f, diversity_ok=bool(res["pass"] and keep(f)),
                                             iou_aligned=aligned_iou(vox(final["text"]), vox(rec["bricks"])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--max_spend", type=float, default=4.0)
    ap.add_argument("--max_tokens", type=int, default=16000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/harness_vision")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "calls.log", "a", encoding="utf-8")
    log = lambda dct: (logf.write(json.dumps(dct) + "\n"), logf.flush())
    from brickexp.data import get_eval_set, load_split, make_eval_subset
    ev = {r["structure_id"] for r in get_eval_set(30)}
    tr = make_eval_subset([r for r in load_split("train") if r["structure_id"] not in ev], per_bucket=35, seed=4242)
    by = {b: [r for r in tr if r["bucket"] == b] for b in ("small", "medium", "large")}
    recs = [x for t in zip(*by.values()) for x in t][:a.n]
    assert not ({r["structure_id"] for r in recs} & ev)
    cl = Client(a.max_tokens, a.max_spend, log)
    jobs = [(r, arm) for r in recs for arm in ("A", "B")]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda ra: episode(cl, *ra, out), jobs))
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res), encoding="utf-8")
    spent = cl.spent()
    S = {}
    for arm in ("A", "B"):
        R = [r for r in res if r["arm"] == arm]; iou = [r["iou_aligned"] for r in R if r["pass"] and "iou_aligned" in r]
        calls = [c for c in cl.calls if c["tag"].startswith(arm)]
        S[arm] = {"n": len(R), "pass": sum(r["pass"] for r in R), "diversity_ok": sum(r.get("diversity_ok", False) for r in R),
                  "iou_aligned_mean_passed": sum(iou) / len(iou) if iou else None, "n_iou": len(iou),
                  "mean_rounds": sum(r["rounds"] for r in R) / len(R), "calls": len(calls),
                  "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
                  "completion_tokens": sum(c["completion_tokens"] or 0 for c in calls),
                  "truncated": sum(c["finish"] == "length" for c in calls)}
    tot = sum(S[x]["prompt_tokens"] + 0 for x in S)
    summ = {"spent_cny_total": spent, "random_human_pair_iou_ref": 0.151, "arms": S}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
