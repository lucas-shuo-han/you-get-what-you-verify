"""DeepSeek black-box distillation pilot (step 1): deepseek-chat on the StableText2Brick prompt, scored by our verify
(format / bounds / collision / floating + Gurobi stability).

One run with max_budget=3 gives both settings: (a) one-shot = attempt 0, (b) feedback harness = solved within <=3
feedback rounds (verify message fed back, history window 2 as in our eval).

Key: read from ~/.deepseek_key inside this process only; never printed, logged or written. Results hold no key.
Cost guard: queries the account balance (GET /user/balance) before every round and stops if the spend so far
would exceed --max_spend (CNY).

python tools/distill_pilot.py --n 3 --out results/distill_pilot/smoke3.jsonl
python tools/distill_pilot.py --n 20 --out results/distill_pilot/p20.jsonl --max_spend 0.5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brickexp.evaluate import run_eval, success_curve, failure_breakdown  # noqa: E402
from brickexp.data import get_eval_set  # noqa: E402

API = "https://api.deepseek.com"
MODEL = "deepseek-chat"


def _key():
    return (Path.home() / ".deepseek_key").read_text().strip()


def balance():
    r = requests.get(f"{API}/user/balance", headers={"Authorization": f"Bearer {_key()}"}, timeout=30)
    r.raise_for_status()
    for b in r.json().get("balance_infos", []):
        if b.get("currency") == "CNY":
            return float(b["total_balance"])
    return None


class DeepSeek:
    def __init__(self, max_tokens, temperature, max_spend, log, api_workers=1, pool=None, quality=False):
        self.api_workers, self.pool, self.quality = api_workers, pool, quality
        self.model = MODEL
        self.max_tokens, self.temperature, self.max_spend, self.log = max_tokens, temperature, max_spend, log
        self.usage = []  # one dict per call: prompt/completion/cache-hit tokens, seconds
        self.b0 = balance()
        log({"event": "start", "balance_cny": self.b0})

    def spent(self):
        b = balance()
        return None if b is None or self.b0 is None else round(self.b0 - b, 4)

    def generate(self, conversations, metas):
        s = self.spent()
        if s is not None and s >= self.max_spend:
            raise SystemExit(f"spend guard: {s} CNY >= {self.max_spend}")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.api_workers) as ex:
            outs = list(ex.map(self._one, conversations, metas))
        if self.pool is not None:  # pre-check all outputs in parallel; run_eval then reads PRECHECK
            from brickexp.checker import extract_bricks_text
            todo = sorted({o for o in outs if extract_bricks_text(o) and o not in PRECHECK})
            for o, r in zip(todo, self.pool.map(staged_check, todo, [self.quality] * len(todo))):
                PRECHECK[o] = r
        return outs

    def _one(self, conv, m):
        if True:
            t = time.time()
            for tries in range(3):
                try:
                    r = requests.post(f"{API}/chat/completions", timeout=300,
                                      headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
                                      json={"model": self.model, "messages": conv, "max_tokens": self.max_tokens,
                                            "temperature": self.temperature,
                                            **({"reasoning_effort": "low"} if self.model == "deepseek-flash" else {})})
                    r.raise_for_status(); break
                except requests.RequestException as e:  # status only; never the request headers
                    self.log({"event": "http_error", "status": getattr(e.response, "status_code", None), "try": tries})
                    time.sleep(5)
            else:
                return ""
            j = r.json(); u = j.get("usage", {})
            rec = {"idx": m.get("structure_id"), "attempt": m["attempt"], "prompt_tokens": u.get("prompt_tokens"),
                   "completion_tokens": u.get("completion_tokens"), "cache_hit": u.get("prompt_cache_hit_tokens"),
                   "finish": j["choices"][0].get("finish_reason"), "sec": round(time.time() - t, 1)}
            self.usage.append(rec); self.log({"event": "call", **rec})
            return j["choices"][0]["message"]["content"] or ""


DIVERSITY = ("\nModel the described shape (its parts, openings and silhouette), not a solid block: "
             "use at least 3 different brick sizes and do not fill the bounding box solid.")


def install_quality_feedback():
    """Physics-ok structures failing the pre-registered quality filter are fed back as a 'quality' failure."""
    import brickexp.evaluate as ev
    from dataclasses import replace
    from distill_filter import feats, keep, TH
    base = ev.check

    def check(text, **kw):
        r = base(text, **kw)
        if r.ok:
            f = feats(text)
            if not keep(f):
                return replace(r, ok=False, failure="quality", message=(
                    f"The structure is physically valid but too simple: {f['n_types']} brick sizes "
                    f"(need >= {TH['n_types']}), largest single-size share {f['max_type_share']:.0%} (need <= "
                    f"{TH['max_type_share']:.0%}), fill of bounding box {f['fill']:.0%} (need <= {TH['fill']:.0%}). "
                    "Rebuild it to follow the described shape with varied brick sizes."))
        return r
    ev.check = check


PRECHECK = {}  # text -> CheckResult, filled in parallel right after each generation round (scale-up mode)


def staged_check(text, quality):
    """cheap checks -> quality filter -> Gurobi (Gurobi only for structures that pass everything else)."""
    from dataclasses import replace
    from brickexp.checker import check
    from distill_filter import feats, keep, TH
    r = check(text, use_gurobi=False)
    if not r.ok:
        return r
    if quality:
        f = feats(text)
        if not keep(f):
            return replace(r, ok=False, failure="quality", message=(
                f"The structure is physically valid but too simple: {f['n_types']} brick sizes "
                f"(need >= {TH['n_types']}), largest single-size share {f['max_type_share']:.0%} (need <= "
                f"{TH['max_type_share']:.0%}), fill of bounding box {f['fill']:.0%} (need <= {TH['fill']:.0%}). "
                "Rebuild it to follow the described shape with varied brick sizes."))
    return check(text, use_gurobi=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_budget", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_spend", type=float, default=0.3)
    ap.add_argument("--skip", type=int, default=0, help="skip the first prompts (already used in the smoke)")
    ap.add_argument("--split", default="eval", choices=["eval", "train"],
                    help="train: prompts from the StableText2Brick TRAIN split, asserted disjoint from the 90 eval prompts")
    ap.add_argument("--api_workers", type=int, default=1)
    ap.add_argument("--model", default="deepseek-chat", choices=["deepseek-chat", "deepseek-flash"])
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--gurobi_workers", type=int, default=0, help=">0: staged parallel checks (cheap -> quality -> Gurobi)")
    ap.add_argument("--quality", action="store_true", help="diversity line in the prompt + quality filter as feedback")
    a = ap.parse_args()
    global MODEL
    MODEL = a.model
    if a.quality:
        import brickexp.prompts as P
        _ins = P.instruction
        P.instruction = lambda c: _ins(c) + DIVERSITY
        install_quality_feedback()
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(out.with_suffix(".log"), "a", encoding="utf-8")
    log = lambda d: (print(json.dumps(d), flush=True), logf.write(json.dumps(d) + "\n"), logf.flush())
    recs = get_eval_set(30)  # same 90 prompts as Ref / C1 (small, medium, large x 30); interleave buckets
    if a.split == "train":
        from brickexp.data import load_split, make_eval_subset
        ev = {r["structure_id"] for r in recs}
        tr = [r for r in load_split("train") if r["structure_id"] not in ev]
        recs = make_eval_subset(tr, per_bucket=(a.skip + a.n + 2) // 3 + 1, seed=a.seed)
        assert not ({r["structure_id"] for r in recs} & ev), "train prompts overlap the eval set"
    by = {b: [r for r in recs if r["bucket"] == b] for b in ("small", "medium", "large")}
    order = [x for trio in zip(*by.values()) for x in trio][a.skip:a.skip + a.n]
    pool = None
    if a.gurobi_workers:
        from concurrent.futures import ProcessPoolExecutor
        import brickexp.evaluate as ev
        pool = ProcessPoolExecutor(a.gurobi_workers)
        _base = ev.check
        ev.check = lambda text, **kw: PRECHECK[text] if text in PRECHECK else _base(text, **kw)
    be = DeepSeek(a.max_tokens, a.temperature, a.max_spend, log, a.api_workers, pool, a.quality)
    try:
        states = run_eval(order, be, max_budget=a.max_budget, use_gurobi=True, history_window=2, out_path=out)
    finally:
        sp = be.spent()
        log({"event": "end", "spent_cny": sp, "calls": len(be.usage),
             "prompt_tokens": sum(u["prompt_tokens"] or 0 for u in be.usage),
             "completion_tokens": sum(u["completion_tokens"] or 0 for u in be.usage),
             "truncated": sum(u["finish"] == "length" for u in be.usage)})
    log({"event": "summary", "curve": success_curve(states, budgets=(0, 1, 2, 3)), "failures": failure_breakdown(states)})


if __name__ == "__main__":
    main()
