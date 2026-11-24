"""Train a tiny decoder-only transformer from scratch on one arm, then evaluate k=0 (and k>0 with feedback).

Token budget is fixed across arms (--train_tokens); arms with fewer tokens per epoch simply repeat.
Usage:
  python -m brickexp.tiny.train --data data/tiny/n4000_s0 --arm RF --seed 0 --out runs/tiny/RF_s0
  --smoke 20  : 20 steps, print peak memory + tokens/s, no eval
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vocab as V
from . import geom_mask as GM


class Block(nn.Module):
    def __init__(self, d, nh, drop):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.nh, self.drop = nh, drop

    def forward(self, x, cache=None, pos=None):
        """cache: (K, V) buffers [B, nh, Lmax, hd] filled up to pos[b] (exclusive); x is one new token per row."""
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.nh, C // self.nh).permute(2, 0, 3, 1, 4)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0)
        else:
            K, Vc = cache
            rows = torch.arange(B, device=x.device)
            K[rows, :, pos] = k[:, :, 0].to(K.dtype); Vc[rows, :, pos] = v[:, :, 0].to(Vc.dtype)
            mask = torch.arange(K.shape[2], device=x.device)[None, :] <= pos[:, None]  # [B, Lmax]
            y = F.scaled_dot_product_attention(q, K.to(q.dtype), Vc.to(q.dtype), attn_mask=mask[:, None, None, :])
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, d=384, n_layer=8, nh=6, ctx=2048, drop=0.0):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V.VOCAB_SIZE, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, nh, drop) for _ in range(n_layer)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V.VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight
        self.ctx = ctx
        self.apply(self._init)  # GPT-2 style; default N(0,1) embeddings + weight tying gave initial loss ~54

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, caches=None, pos=None):
        if caches is None:
            x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
            for b in self.blocks:
                x = b(x)
        else:  # one token per row at per-row position pos
            x = self.tok(idx) + self.pos(pos)[:, None]
            for b, c in zip(self.blocks, caches):
                x = b(x, c, pos)
        return self.head(self.ln(x))


def batches(seqs, bs, ctx, rng):
    """Sequence-per-row batches (pad to longest, truncate to ctx). Loss on non-PAD targets; v2 masks bad bricks in main()."""
    order = rng.permutation(len(seqs))
    for i in range(0, len(order) - bs + 1, bs):
        rows = [seqs[j][:ctx + 1] for j in order[i:i + bs]]
        L = max(map(len, rows))
        arr = np.full((bs, L), V.PAD, dtype=np.int64)
        for r, s in enumerate(rows):
            arr[r, :len(s)] = s
        yield torch.from_numpy(arr)


def mask_bad_bricks(x: torch.Tensor) -> torch.Tensor:
    """DIRECTION §4d: copy of a token batch with injected bad-brick tokens set to PAD (=> no loss as targets).
    A bad brick is the 4-token brick (T X Y Z) immediately followed by FB (R+F, R+F~) or by BACK (R).
    FB, failure type, digits, BACK and the correction brick keep their loss. No-op on S (no FB/BACK)."""
    y = x.clone()
    is_t = torch.zeros(V.VOCAB_SIZE, dtype=torch.bool)
    is_t[[V.ID[f"T_{b}"] for b in V.BRICK_TYPES]] = True
    is_t = is_t.to(x.device)
    trig = (x[:, 4:] == V.FB) | (x[:, 4:] == V.BACK)  # trigger at p, brick at p-4..p-1
    start = trig & is_t[x[:, :-4]]
    for o in range(4):
        y[:, o:o + start.shape[1]][start] = V.PAD
    return y


def _pick(last, temperature, gen):
    last[:, V.PAD] = -1e9; last[:, V.BOS] = -1e9
    if temperature <= 0:
        return last.argmax(-1).tolist()
    return torch.multinomial(F.softmax(last / temperature, -1), 1, generator=gen).squeeze(1).tolist()


@torch.no_grad()
def generate_nocache(model, prefixes, max_new, temperature, device, gen=None, geom=False):
    """Reference implementation (re-runs the full prefix every step). Stops a row at EOS or ctx."""
    model.eval()
    out = [list(p) for p in prefixes]
    done = [False] * len(out)
    trk = [GM.GeomTracker(p) for p in prefixes] if geom else None
    for _ in range(max_new):
        live = [i for i, d in enumerate(done) if not d]
        if not live:
            break
        L = max(len(out[i]) for i in live)
        arr = torch.full((len(live), L), V.PAD, dtype=torch.long)
        lens = []
        for r, i in enumerate(live):
            s = out[i][-model.ctx:]
            arr[r, :len(s)] = torch.tensor(s); lens.append(len(s))
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(arr.to(device))
        last = logits[torch.arange(len(live)), torch.tensor(lens) - 1].float()
        if trk:
            last = GM.apply(last, [trk[i] for i in live])
        nxt = _pick(last, temperature, gen)
        for r, i in enumerate(live):
            out[i].append(nxt[r])
            if trk:
                trk[i].update(nxt[r])
            if nxt[r] == V.EOS or len(out[i]) >= model.ctx:
                done[i] = True
    return out


@torch.no_grad()
def generate(model, prefixes: list[list[int]], max_new: int, temperature: float, device, gen=None,
             geom=False) -> list[list[int]]:
    """Batched sampling with a per-row KV cache (rows have different prefix lengths; per-row positions + mask).
    Same stopping rule as generate_nocache (EOS or ctx); token-identical to it under greedy (tests)."""
    model.eval()
    B = len(prefixes)
    if B == 0:
        return []
    out = [list(p[-model.ctx:]) for p in prefixes]
    trk = [GM.GeomTracker(p) for p in prefixes] if geom else None
    nh = model.blocks[0].nh
    hd = model.tok.weight.shape[1] // nh
    dt = torch.float16 if device.type == "cuda" else torch.float32
    Lmax = min(model.ctx, max(map(len, out)) + max_new)
    caches = [(torch.zeros(B, nh, Lmax, hd, device=device, dtype=dt), torch.zeros(B, nh, Lmax, hd, device=device, dtype=dt))
              for _ in model.blocks]
    # prefill: full causal pass on the right-padded batch, copy K/V of real positions into the cache
    lens = torch.tensor([len(s) for s in out], device=device)
    arr = torch.full((B, int(lens.max())), V.PAD, dtype=torch.long)
    for r, s in enumerate(out):
        arr[r, :len(s)] = torch.tensor(s)
    arr = arr.to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        x = model.tok(arr) + model.pos(torch.arange(arr.shape[1], device=device))
        for b, (K, Vc) in zip(model.blocks, caches):
            Bq, T, C = x.shape
            q, k, v = b.qkv(b.ln1(x)).view(Bq, T, 3, nh, hd).permute(2, 0, 3, 1, 4)
            K[:, :, :T] = k.to(dt); Vc[:, :, :T] = v.to(dt)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + b.proj(y.transpose(1, 2).reshape(Bq, T, C))
            x = x + b.mlp(b.ln2(x))
        logits = model.head(model.ln(x))
    last = logits[torch.arange(B, device=device), lens - 1].float()
    done = [False] * B
    pos = lens.clone()  # next write position per row (cache row r <-> rows[r])
    rows = list(range(B))  # output index of each cache row; cache is compacted in place only when half are done
    live = list(range(B))
    for step in range(max_new):
        if trk:
            last = GM.apply(last, [trk[i] for i in live])
        nxt = _pick(last, temperature, gen)  # one row per live output, in order (same as generate_nocache)
        for r, i in enumerate(live):
            out[i].append(nxt[r])
            if trk:
                trk[i].update(nxt[r])
            if nxt[r] == V.EOS or len(out[i]) >= model.ctx:
                done[i] = True
        live = [i for i in range(B) if not done[i]]
        if not live or step == max_new - 1:
            break
        if len(live) <= len(rows) // 2:  # compact: drop finished rows (one copy, amortised)
            keep = torch.tensor([r for r, i in enumerate(rows) if not done[i]], device=device)
            caches = [(K[keep], Vc[keep]) for K, Vc in caches]
            pos = pos[keep]
            rows = [rows[r] for r in keep.tolist()]
            del keep
        tok = torch.tensor([out[i][-1] if not done[i] else V.PAD for i in rows], device=device)[:, None]
        p = pos.clamp(max=caches[0][0].shape[2] - 1)  # finished rows: harmless write, output ignored
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            lg = model(tok, caches, p)
        pos += 1
        sel = torch.tensor([r for r, i in enumerate(rows) if not done[i]], device=device)
        last = lg[sel, 0].float()
    # restore any prefix that was longer than ctx (not truncated in the no-cache reference either: it keeps full out)
    return [list(p[:-model.ctx]) + o if len(p) > model.ctx else o for p, o in zip(prefixes, out)]


def evaluate(model, cats: list[int], device, max_budget: int, fb_mode: str, use_gurobi, temperature=0.8,
             max_new=1800, eval_bs=64, geom=False):
    """k-retry eval. On failure: keep tokens up to the bad brick, append feedback (fb_mode 'true' for F arms,
    'none' for R/S) + BACK, continue. Returns per-prompt dicts."""
    from ..checker import check
    from .per_brick_metrics import brick_stats
    states = [{"cat": c, "ids": [V.BOS, V.ID[f"CAT_{c}"]], "solved_at": None, "attempts": []} for c in cats]
    active = list(range(len(states)))
    t_eval = time.time()
    for k in range(max_budget + 1):
        if not active:
            break
        print(f"[eval] k={k} active={len(active)} elapsed={time.time() - t_eval:.0f}s", flush=True)
        outs = []
        for c in range(0, len(active), eval_bs):  # chunked: KV cache memory ~ eval_bs x layers x Lmax
            outs += generate(model, [states[i]["ids"] for i in active[c:c + eval_bs]], max_new, temperature, device,
                             geom=geom)
        still = []
        n_pre = {i: len(states[i]["ids"]) for i in active}
        for i, ids in zip(active, outs):
            bricks, st = V.decode(ids)
            res = check(V.bricks_to_text(bricks), use_gurobi=use_gurobi) if bricks else None
            ok = bool(res and res.ok)
            pb = brick_stats(V.bricks_to_text(bricks)) if bricks else {"n_floating": 0, "first_error": 0}
            states[i]["attempts"].append({"k": k, "n_bricks": len(bricks), "n_floating": pb["n_floating"],
                                          "first_error": pb["first_error"], "failure": None if ok else
                                          (res.failure if res else "empty"), **st,
                                          "text": V.bricks_to_text(bricks),
                                          # raw generated token stream (after the prompt/retry context), incl. BACK
                                          "raw": " ".join(V.TOKENS[t] for t in ids[n_pre[i]:]),
                                          "raw_prefix_len": n_pre[i]})
            if ok:
                states[i]["solved_at"] = k
                continue
            still.append(i)
            # rebuild context: resolved prefix up to and including the bad brick, then feedback + BACK
            bad = res.first_bad_index if (res and res.first_bad_index is not None) else len(bricks) - 1
            ctx = [V.BOS, V.ID[f"CAT_{states[i]['cat']}"]]
            for b in bricks[:max(bad, 0) + 1]:
                ctx += V.brick_tokens(b)
            if res and res.failure in V.FAIL_TYPES and fb_mode == "true":
                ctx += V.feedback_tokens(res.failure, bad + 1)
            if bricks:
                ctx.append(V.BACK)
            states[i]["ids"] = ctx
        active = still
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arm", required=True, choices=["S", "RF", "RFs", "R"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=1536)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--train_tokens", type=float, default=0, help="token budget; 0 = 10 epochs of the R+F arm")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--eval_per_cat", type=int, default=20)
    ap.add_argument("--eval_k", type=int, default=10)
    ap.add_argument("--checker", default="connectivity", choices=["connectivity", "gurobi"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--eval_bs", type=int, default=64, help="prompts per generate() call (KV-cache memory)")
    ap.add_argument("--geom_mask", action="store_true", help="§4a: decode-time collision/out-of-bounds masking")
    ap.add_argument("--eval_tag", default=None, help="suffix for eval outputs (default 'geom' with --geom_mask)")
    ap.add_argument("--eval_model", default=None, help="model.pt to load with --eval_only (default <out>/model.pt)")
    ap.add_argument("--eval_only", action="store_true", help="load <out>/model.pt and run only eval")
    ap.add_argument("--loss_mask", default="bad", choices=["bad", "none"],
                    help="§4d: 'bad' = no loss on injected bad-brick tokens (v2); 'none' = v1 behaviour")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    data = Path(a.data)
    seqs = list(np.load(data / f"{a.arm}.npy", allow_pickle=True))
    stats = json.loads((data / "stats.json").read_text())
    budget = a.train_tokens or 10 * stats["tokens"]["RF"]
    model = TinyGPT(a.d, a.layers, a.heads, a.ctx).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    tok_per_step = a.bs * float(np.mean([min(len(s), a.ctx + 1) for s in seqs]))
    total_steps = a.smoke or int(budget / tok_per_step)
    warm = max(1, total_steps // 50)
    sched = lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total_steps)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"arm={a.arm} params={n_params/1e6:.1f}M seqs={len(seqs)} steps={total_steps} device={device}", flush=True)
    if a.eval_only:
        model.load_state_dict(torch.load(a.eval_model or out / "model.pt", map_location=device))
        tl = out / "train_log.json"
        prev = json.loads(tl.read_text()) if tl.exists() else {"summary": {}, "log": []}
        summary, log = prev["summary"], prev["log"]
        print("eval_only: loaded", a.eval_model or out / "model.pt", flush=True)
    else:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0, step, seen, log = time.time(), 0, 0, []
        model.train()
        while step < total_steps:
            for x in batches(seqs, a.bs, a.ctx, rng):
                x = x.to(device)
                inp, tgt = x[:, :-1], x[:, 1:]
                if a.loss_mask == "bad" and a.arm != "S":  # §4d; S path untouched (identical to v1)
                    tgt = mask_bad_bricks(x)[:, 1:]
                for g in opt.param_groups:
                    g["lr"] = a.lr * sched(step)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(inp)
                loss = F.cross_entropy(logits.float().reshape(-1, V.VOCAB_SIZE), tgt.reshape(-1), ignore_index=V.PAD)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                seen += int((tgt != V.PAD).sum())
                step += 1
                if step % 50 == 0 or step == total_steps or a.smoke:
                    el = time.time() - t0
                    log.append({"step": step, "loss": loss.item(), "tok": seen, "sec": el})
                    if step % 50 == 0 or step == total_steps:
                        print(f"step {step}/{total_steps} loss {loss.item():.3f} tok/s {seen / el:.0f}", flush=True)
                if step >= total_steps:
                    break
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
        summary = {"args": vars(a), "params": n_params, "steps": step, "train_tokens": seen,
                   "train_seconds": time.time() - t0, "tok_per_s": seen / (time.time() - t0), "peak_mem_gb": peak,
                   "final_loss": log[-1]["loss"] if log else None}
        print("PEAK_MEM_GB", f"{peak:.2f}", "TOK_PER_S", f"{summary['tok_per_s']:.0f}", flush=True)
        (out / "train_log.json").write_text(json.dumps({"summary": summary, "log": log}, indent=1))
        if a.smoke:
            return
        torch.save(model.state_dict(), out / "model.pt")
    cats = [c for c in range(len(stats["category_map"])) for _ in range(a.eval_per_cat)]
    fb_mode = "true" if a.arm in ("RF", "RFs") else "none"
    t1 = time.time()
    torch.manual_seed(a.seed + 1000)  # eval sampling RNG independent of training -> --eval_only reproduces a full run
    states = evaluate(model, cats, device, a.eval_k, fb_mode, use_gurobi=a.checker == "gurobi",
                      eval_bs=a.eval_bs, geom=a.geom_mask)
    tag = a.eval_tag if a.eval_tag is not None else ("geom" if a.geom_mask else "")
    sfx = f"_{tag}" if tag else ""
    curve = {k: float(np.mean([s["solved_at"] is not None and s["solved_at"] <= k for s in states]))
             for k in (0, 1, 10) if k <= a.eval_k}
    with open(out / f"eval{sfx}.jsonl", "w", encoding="utf-8") as f:
        for s in states:
            f.write(json.dumps({k: v for k, v in s.items() if k != "ids"}) + "\n")
    from .per_brick_metrics import bootstrap
    pb = bootstrap([{"n_placed": s["attempts"][0]["n_bricks"], "n_floating": s["attempts"][0]["n_floating"],
                     "first_error": s["attempts"][0]["first_error"]} for s in states])
    print("PER_BRICK_K0", json.dumps(pb), flush=True)
    summary.update({f"per_brick_k0{sfx}": pb, f"eval_curve{sfx}": curve, f"eval_seconds{sfx}": time.time() - t1, "eval_checker": a.checker,
                    f"eval_geom_mask{sfx}": a.geom_mask})
    (out / "train_log.json").write_text(json.dumps({"summary": summary, "log": log}, indent=1))
    print("EVAL", json.dumps(curve), flush=True)


if __name__ == "__main__":
    main()
