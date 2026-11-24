"""Audit (read-only): at k=0, how often does the model self-emit FB / BACK, and are floating kept bricks the
ones placed right after a self-emitted BACK?  Run from repo root:
    python audit_self_back.py runs/RF_s1/eval.jsonl runs/S_s1/eval.jsonl runs/R_s1/eval.jsonl runs/RFs_s1/eval.jsonl
Uses k=0 attempts only (raw_prefix_len == 2, so raw is the whole stream). Mirrors vocab.decode exactly.
"""
import json, sys
from brickexp.tiny.per_brick_metrics import floating_flags, post_filter


def walk(names):
    """-> kept bricks [(h,w,x,y,z,after_back)], n_fb, n_back, n_back_popping_nothing"""
    kept, i, n_fb, n_back, n_empty, after_back = [], 0, 0, 0, 0, False
    while i < len(names):
        t = names[i]
        if t in ("EOS", "PAD"):
            break
        if t == "BACK":
            n_back += 1
            if kept: kept.pop()
            else: n_empty += 1
            after_back = True; i += 1
        elif t == "FB":
            n_fb += 1; i += 5
        elif t.startswith("T_") and i + 3 < len(names) and names[i+1][0] == "X" and names[i+2][0] == "Y" and names[i+3][0] == "Z":
            h, w = map(int, t[2:].split("x"))
            kept.append((h, w, int(names[i+1][1:]), int(names[i+2][1:]), int(names[i+3][1:]), after_back))
            after_back = False; i += 4
        else:
            i += 1
    return kept, n_fb, n_back, n_empty


for path in sys.argv[1:]:
    n = with_fb = with_back = fb_tok = back_tok = empty = 0
    flo = flo_ab = tot_ab = tot = 0
    for line in open(path):
        r = json.loads(line)
        a = [x for x in r["attempts"] if x["k"] == 0]
        if not a or "raw" not in a[0]:
            continue
        a = a[0]; n += 1
        kept, nf, nb, ne = walk(a["raw"].split())
        with_fb += nf > 0; with_back += nb > 0; fb_tok += nf; back_tok += nb; empty += ne
        # floating on the post-filtered kept sequence (primary §4c metric, mixed rule)
        from brickgpt.data.brick_structure import Brick
        bricks = [Brick(h=h, w=w, x=x, y=y, z=z) for h, w, x, y, z, _ in kept]
        kb, _, _ = post_filter(bricks)
        ids_kept = {id(b) for b in kb}
        sel = [j for j, b in enumerate(bricks) if id(b) in ids_kept]
        fl = floating_flags(kb, "mixed"); ab = [kept[j][5] for j in sel]
        for f, b in zip(fl, ab):
            tot += 1; flo += f; tot_ab += b; flo_ab += f and b
    print(json.dumps({"file": path, "prompts_k0": n, "frac_prompts_with_FB": with_fb / max(n, 1),
                      "frac_prompts_with_BACK": with_back / max(n, 1), "FB_per_prompt": fb_tok / max(n, 1),
                      "BACK_per_prompt": back_tok / max(n, 1), "BACK_on_empty": empty,
                      "kept_bricks": tot, "floating": flo, "floating_frac": flo / max(tot, 1),
                      "kept_after_selfBACK": tot_ab, "floating_after_selfBACK": flo_ab,
                      "frac_floating_that_follow_selfBACK": flo_ab / max(flo, 1),
                      "floating_rate_after_BACK": flo_ab / max(tot_ab, 1),
                      "floating_rate_otherwise": (flo - flo_ab) / max(tot - tot_ab, 1)}))
