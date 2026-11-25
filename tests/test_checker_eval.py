import pytest

from brickexp import checker
from brickexp.backends import ReplayBackend
from brickexp.data import SPLIT_DIR, bucket_of, load_jsonl, n_bricks
from brickexp.evaluate import build_conversation, failure_breakdown, run_eval, success_curve

C = dict(use_gurobi=False)  # fast connectivity path; Gurobi tested separately


def test_ok_simple():
    r = checker.check("2x4 (0,0,0)\n2x4 (0,2,1)\n", **C)
    assert r.ok and r.failure is None and r.n_bricks == 2


def test_format():
    r = checker.check("hello world", **C)
    assert not r.ok and r.failure == "format"
    assert checker.check("2x4 (0,0,0)\nbanana\n", strict=True, **C).failure == "format"
    assert checker.check("3x3 (0,0,0)", **C).failure == "format"  # unsupported size


def test_tolerates_prose_and_fences():
    r = checker.check("Here you go:\n```\n2x4 (0,0,0)\n```\n", **C)
    assert r.ok and r.n_bricks == 1


def test_collision():
    r = checker.check("2x4 (0,0,0)\n2x4 (0,2,0)\n", **C)
    assert r.failure == "collision" and r.first_bad_index == 1


def test_out_of_bounds():
    r = checker.check("2x4 (0,0,0)\n2x4 (19,0,0)\n", **C)
    assert r.failure == "out_of_bounds" and r.first_bad_index == 1


def test_floating():
    r = checker.check("2x4 (0,0,0)\n2x4 (5,5,3)\n", **C)
    assert r.failure == "floating" and r.first_bad_index == 1


def test_disconnected_unstable_connectivity():
    # B at z=1 and C at z=2 support each other (not floating) but neither reaches the ground.
    r = checker.check("2x2 (0,0,0)\n2x2 (5,5,1)\n2x2 (5,5,2)\n", **C)
    assert r.failure == "unstable" and r.first_bad_index == 1 and r.method == "connectivity"


CANTILEVER = "1x1 (0,0,0)\n8x1 (0,0,1)\n" + "".join(f"6x2 (7,0,{z})\n" for z in range(2, 20))


@pytest.mark.skipif(not checker.gurobi_available(), reason="no Gurobi license")
def test_gurobi_cantilever_unstable():
    assert checker.check(CANTILEVER, use_gurobi=False).ok  # connectivity can't see it
    r = checker.check(CANTILEVER, use_gurobi=True)
    assert r.method == "gurobi" and r.failure == "unstable"
    assert checker.check("2x4 (0,0,0)\n2x4 (0,2,1)\n", use_gurobi=True).ok


def test_buckets():
    assert bucket_of(10) == "small" and bucket_of(61) == "medium" and bucket_of(121) == "large"
    assert n_bricks("1x1 (0,0,0)\n\n1x1 (0,0,1)\n") == 2


def test_conversation_window():
    hist = [{"text": f"t{i}", "message": f"m{i}"} for i in range(5)]
    conv = build_conversation("a chair", hist, history_window=2)
    assert len(conv) == 2 + 4 and conv[2]["content"] == "t3"
    assert len(build_conversation("a chair", hist, 0)) == 2


def _eval_records():
    p = SPLIT_DIR / "test_eval_10_s0.jsonl"
    if not p.exists():
        pytest.skip("run `python -m brickexp.data --per_bucket 10` first")
    return load_jsonl(p)


def test_replay_all_succeed():
    recs = _eval_records()
    states = run_eval(recs, ReplayBackend(), max_budget=0, verbose=False, **C)
    curve = success_curve(states)
    assert curve["all"][0] == 1.0
    assert set(curve) == {"all", "small", "medium", "large"}


def test_replay_retry_path(tmp_path):
    recs = _eval_records()[:6]
    out = tmp_path / "traj.jsonl"
    states = run_eval(recs, ReplayBackend(fail_first=1), max_budget=3, out_path=out, verbose=False, **C)
    curve = success_curve(states)
    assert curve["all"][0] == 0.0 and curve["all"][1] == 1.0
    assert failure_breakdown(states) == {"floating": 6, "ok": 6}
    rows = load_jsonl(out)
    assert len(rows) == 6 and all(len(r["attempts"]) == 2 and r["solved_at"] == 1 for r in rows)


def test_budget_zero_never_retries():
    recs = _eval_records()[:3]
    states = run_eval(recs, ReplayBackend(fail_first=5), max_budget=0, verbose=False, **C)
    assert all(len(s["attempts"]) == 1 and s["solved_at"] is None for s in states)


def test_gurobi_required_raises(monkeypatch):
    monkeypatch.setattr(checker, "_gurobi_ok", False)
    with pytest.raises(checker.GurobiUnavailable):
        checker.check("2x4 (0,0,0)\n", use_gurobi=True)
    assert checker.check("2x4 (0,0,0)\n", use_gurobi=False).ok


def test_analysis_ci(tmp_path):
    from brickexp.analysis import summarize, plot, error_locality
    recs = _eval_records()
    a = run_eval(recs, ReplayBackend(fail_first=1), max_budget=1, verbose=False, **C)
    b = run_eval(recs, ReplayBackend(fail_first=2), max_budget=1, verbose=False, **C)
    s = summarize([a, b])
    assert s["all"][0]["mean"] == 0 and s["all"][1]["mean"] == 0.5 and s["all"][1]["lo"] <= 0.5 <= s["all"][1]["hi"]
    assert error_locality([a])[0]["local"] == len(recs)
    pytest.importorskip("matplotlib")
    assert (tmp_path / "f.png").exists() or plot({"x": s}, str(tmp_path / "f.png"))


# Real Kaggle v6 false positive: one space-separated line with '[' prefix; overlapping 2x4s.
V6_ONE_LINE = ("[2x4 (0,0,0) 2x4 (1,0,0) 2x4 (2,0,0) 1x2 (0,1,0) 2x2 (1,1,0) 2x2 (2,1,0) "
               "1x4 (0,2,0) 2x2 (1,2,0) 2x2 (2,2,0) 2x2 (3,2,0)")


def test_v6_one_line_output_is_not_valid():
    from brickexp.checker import check
    r = check(V6_ONE_LINE, use_gurobi=False)
    assert not r.ok
    assert r.failure == "collision"
    assert r.n_bricks == 10


def test_chunked_ce_mixed_dtype():
    # Kaggle C1 v10: fp16 AMP gave fp32 hidden states with an fp16 lm_head -> matmul dtype error
    import torch
    from brickexp.train_sft import chunked_ce
    torch.manual_seed(0)
    h = torch.randn(1, 40, 16, dtype=torch.float32, requires_grad=True)
    w = torch.randn(50, 16).half()
    y = torch.randint(0, 50, (1, 40)); y[:, :10] = -100
    loss = chunked_ce(h, w, y, chunk=7)
    ref = torch.nn.functional.cross_entropy((h[:, :-1].reshape(-1, 16).half().float() @ w.float().t()),
                                            y[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.isfinite(loss) and abs(loss.item() - ref.item()) < 1e-2
    loss.backward()
    assert h.grad is not None


def test_tiny_eval_retry_path(monkeypatch):
    import torch
    from brickgpt.data.brick_structure import Brick
    from brickexp.tiny import train as T, vocab as V
    good = [Brick(h=2, w=4, x=0, y=0, z=0), Brick(h=2, w=4, x=0, y=0, z=1)]
    floating = [good[0], Brick(h=1, w=1, x=9, y=9, z=5)]
    seen = []

    def fake_generate(model, prefixes, max_new, temperature, device, **kw):
        seen.append([list(p) for p in prefixes])
        bricks = floating if len(seen) == 1 else good[1:]
        return [p + sum((V.brick_tokens(b) for b in bricks), []) + [V.EOS] for p in prefixes]

    monkeypatch.setattr(T, "generate", fake_generate)
    st = T.evaluate(None, [3], torch.device("cpu"), 2, "true", use_gurobi=False)[0]
    assert st["attempts"][0]["failure"] == "floating" and st["solved_at"] == 1
    ctx = seen[1][0]  # retry context: CAT, both bricks up to the bad one, feedback(floating, 2), BACK
    assert ctx[-6:] == V.feedback_tokens("floating", 2) + [V.BACK]
    assert V.decode(ctx)[0] == good[:1]
    st = T.evaluate(None, [3], torch.device("cpu"), 2, "none", use_gurobi=False)[0]
    assert seen[-1][0][-1] == V.BACK and V.FB not in seen[-1][0]


def _tiny_model_and_prefixes():
    import torch
    from brickexp.tiny import train as T, vocab as V
    torch.manual_seed(0)
    m = T.TinyGPT(d=64, n_layer=2, nh=4, ctx=96).eval()
    for p in m.parameters():  # larger weights -> peaked, non-trivial distributions
        p.data.mul_(5)
    g = torch.Generator().manual_seed(1)
    pre = [[V.BOS, V.ID["CAT_3"]] + torch.randint(5, V.VOCAB_SIZE, (n,), generator=g).tolist() for n in (0, 7, 19, 3)]
    return T, m, pre


def test_tiny_kv_cache_greedy_identical():
    import torch
    T, m, pre = _tiny_model_and_prefixes()
    a = T.generate_nocache(m, pre, 60, 0.0, torch.device("cpu"))
    b = T.generate(m, pre, 60, 0.0, torch.device("cpu"))
    assert a == b and any(len(x) > len(p) + 5 for x, p in zip(a, pre))


def test_tiny_kv_cache_seeded_sampling_identical():
    import torch
    T, m, pre = _tiny_model_and_prefixes()
    a = T.generate_nocache(m, pre, 200, 1.0, torch.device("cpu"), gen=torch.Generator().manual_seed(7))
    b = T.generate(m, pre, 200, 1.0, torch.device("cpu"), gen=torch.Generator().manual_seed(7))
    assert a == b
    assert any(len(x) >= m.ctx for x in a) or all(x[-1] == T.V.EOS for x in a)  # hit ctx or EOS stop paths


def test_tiny_eval_bs_chunking_identical(monkeypatch):
    import torch
    T, m, _ = _tiny_model_and_prefixes()
    import brickexp.checker as C

    class R:  # every attempt fails -> exercises retries with growing, ragged contexts
        ok, failure, first_bad_index = False, "floating", 0
    monkeypatch.setattr(C, "check", lambda *a, **k: R())
    cats = [c % 21 for c in range(9)]
    runs = []
    for bs in (2, 420):
        torch.manual_seed(0)
        st = T.evaluate(m, cats, torch.device("cpu"), 2, "true", use_gurobi=False, temperature=0.0, max_new=40, eval_bs=bs)
        runs.append([[a["text"] for a in s["attempts"]] for s in st])
    assert runs[0] == runs[1]


def test_tiny_geom_mask_never_collides_or_oob():
    import torch
    from brickgpt.data.brick_structure import Brick
    from brickexp.checker import check
    from brickexp.tiny import train as T, vocab as V
    torch.manual_seed(0)
    m = T.TinyGPT(d=64, n_layer=2, nh=4, ctx=400).eval()
    base = [V.BOS, V.ID["CAT_2"]] + V.brick_tokens(Brick(h=2, w=4, x=0, y=0, z=0)) + V.brick_tokens(
        Brick(h=4, w=2, x=16, y=0, z=0))
    retry = base + V.brick_tokens(Brick(h=2, w=4, x=0, y=0, z=0)) + V.feedback_tokens("collision", 3) + [V.BACK]
    pre = [base, retry, [V.BOS, V.ID["CAT_5"]]] * 4
    n_bricks = 0
    for seed in range(3):
        outs = T.generate(m, pre, 380, 1.5, torch.device("cpu"), gen=torch.Generator().manual_seed(seed), geom=True)
        for o in outs:
            bricks, _ = V.decode(o)
            n_bricks += len(bricks)
            if bricks:
                r = check(V.bricks_to_text(bricks), use_gurobi=False)
                assert r.failure not in ("collision", "out_of_bounds", "format"), (r.failure, r.message)
    assert n_bricks > 100  # the mask was actually exercised


def test_tiny_geom_mask_cache_matches_nocache_and_off_is_unchanged():
    import torch
    T, m, pre = _tiny_model_and_prefixes()
    cpu = torch.device("cpu")
    a = T.generate_nocache(m, pre, 120, 1.0, cpu, gen=torch.Generator().manual_seed(3), geom=True)
    b = T.generate(m, pre, 120, 1.0, cpu, gen=torch.Generator().manual_seed(3), geom=True)
    assert a == b
    c = T.generate(m, pre, 120, 1.0, cpu, gen=torch.Generator().manual_seed(3))
    d = T.generate(m, pre, 120, 1.0, cpu, gen=torch.Generator().manual_seed(3), geom=False)
    assert c == d


def test_per_brick_stats_and_bootstrap(tmp_path):
    import json
    from brickexp.tiny.per_brick_metrics import brick_stats, bootstrap, score_file
    # ground, stacked, floating (z=5), ground, floating (z=9)
    txt = "2x4 (0,0,0)\n2x4 (0,0,1)\n1x1 (9,9,5)\n2x2 (5,5,0)\n1x1 (15,15,9)"
    r = brick_stats(txt)
    assert (r["n_placed"], r["n_floating"], r["first_error"], r["floating_idx"]) == (5, 2, 2, [2, 4])
    ok = brick_stats("2x4 (0,0,0)\n2x4 (0,0,1)")
    assert (ok["n_floating"], ok["first_error"]) == (0, 2)
    hanging = brick_stats("2x4 (0,0,0)\n2x4 (5,5,2)\n2x4 (5,5,1)")  # incremental: mid-air when placed -> floating
    assert (hanging["n_floating"], hanging["first_error"]) == (1, 1)  # (5,5,1) is then held from above
    assert brick_stats("") == {"n_placed": 0, "n_floating": 0, "first_error": 0, "floating_idx": []}
    b = bootstrap([r, ok] * 20, n_boot=2000)
    assert abs(b["floating_frac"] - 40 / 140) < 1e-9 and b["first_error"] == 2.0
    lo, hi = b["floating_frac_ci95"]; assert lo <= b["floating_frac"] <= hi and hi - lo > 0
    f = tmp_path / "eval_geom.jsonl"
    f.write_text("\n".join(json.dumps({"attempts": [{"k": 0, "text": t}, {"k": 1, "text": ""}]}) for t in (txt, "")))
    s = score_file(tmp_path, k=0, n_boot=500)
    assert s["n_prompts"] == 2 and s["n_empty"] == 1 and abs(s["floating_frac"] - 0.4) < 1e-9 and s["first_error"] == 1.0


def test_tiny_eval_records_per_brick(monkeypatch):
    import torch
    from brickgpt.data.brick_structure import Brick
    from brickexp.tiny import train as T, vocab as V
    bricks = [Brick(h=2, w=4, x=0, y=0, z=0), Brick(h=1, w=1, x=9, y=9, z=5)]
    monkeypatch.setattr(T, "generate", lambda m, pre, *a, **k: [p + sum((V.brick_tokens(b) for b in bricks), []) + [V.EOS]
                                                                for p in pre])
    st = T.evaluate(None, [1], torch.device("cpu"), 0, "none", use_gurobi=False)[0]
    a = st["attempts"][0]
    assert (a["n_bricks"], a["n_floating"], a["first_error"], a["failure"]) == (2, 1, 1, "floating")


def test_tiny_eval_raw_stream_saved_and_consistent(monkeypatch):
    import torch
    from brickgpt.data.brick_structure import Brick
    from brickexp.tiny import train as T, vocab as V
    b1, b2, bad = Brick(h=2, w=4, x=0, y=0, z=0), Brick(h=2, w=4, x=0, y=0, z=1), Brick(h=1, w=1, x=9, y=9, z=5)
    gen = V.brick_tokens(b1) + V.brick_tokens(bad) + [V.BACK] + V.brick_tokens(b2) + [V.EOS]
    monkeypatch.setattr(T, "generate", lambda m, pre, *a, **k: [p + gen for p in pre])
    a = T.evaluate(None, [4], torch.device("cpu"), 0, "none", use_gurobi=False)[0]["attempts"][0]
    assert a["raw"].split() == [V.TOKENS[t] for t in gen] and "BACK" in a["raw"] and a["raw_prefix_len"] == 2
    assert V.decode([V.BOS, V.ID["CAT_4"]] + [V.ID[t] for t in a["raw"].split()])[0] == [b1, b2]
    assert a["text"] == V.bricks_to_text([b1, b2])


def test_post_filter_and_support_rules(tmp_path):
    import json
    from brickexp.tiny.per_brick_metrics import post_filter_stats, score_file
    txt = "\n".join([
        "2x4 (0,0,0)",      # 0 kept, ground
        "2x4 (0,0,0)",      # collides with kept 0 -> dropped
        "4x2 (18,0,0)",     # out of bounds -> dropped
        "garbage",          # unparseable -> dropped
        "2x4 (5,5,2)",      # kept 1: hanging; held from above by a LATER brick -> mixed ok, incremental floating
        "2x4 (5,5,3)",      # kept 2: earlier brick below -> supported
        "2x4 (5,5,1)",      # kept 3: nothing below (z=0 empty), earlier brick above -> supported in all rules
        "1x1 (15,15,9)",    # kept 4: floating in all rules
    ])
    m = post_filter_stats(txt, "mixed")
    assert (m["n_generated"], m["n_dropped"], m["n_dropped_oob"], m["n_dropped_collision"], m["n_unparseable"]) == (8, 3, 1, 1, 1)
    assert (m["n_placed"], m["n_floating"], m["first_error"]) == (5, 1, 4)
    i = post_filter_stats(txt, "incremental")
    assert (i["n_floating"], i["first_error"]) == (2, 1)
    f = post_filter_stats(txt, "final")
    assert (f["n_floating"], f["first_error"]) == (1, 4)
    (tmp_path / "eval.jsonl").write_text(json.dumps({"attempts": [{"k": 0, "text": txt}]}) + "\n"
                                         + json.dumps({"attempts": [{"k": 0, "text": ""}]}))
    s = score_file(tmp_path, k=0, n_boot=300, post_filter_rule="mixed")
    assert s["mode"] == "post_filter:mixed" and abs(s["floating_frac"] - 0.2) < 1e-9 and abs(s["filtered_out_frac"] - 3 / 8) < 1e-9


# ---- DIRECTION §4d: bad-brick loss mask + R token match ----
_D4 = "data/tiny/n4000_s0"


def _d4_seqs(arm):
    import numpy as np, os, pytest
    if not os.path.exists(f"{_D4}/{arm}.npy"):
        pytest.skip("tiny data not present")
    return list(np.load(f"{_D4}/{arm}.npy", allow_pickle=True))


def test_d4_mask_is_noop_on_S_and_S_loss_unchanged():
    import torch, numpy as np, torch.nn.functional as F
    from brickexp.tiny import train as T, vocab as V
    seqs = _d4_seqs("S")
    for s in seqs:
        x = torch.tensor(s.astype("int64"))[None]
        assert torch.equal(T.mask_bad_bricks(x), x)
    torch.manual_seed(0)
    m = T.TinyGPT(32, 1, 2, 256).eval()
    x = next(T.batches(seqs, 4, 256, np.random.default_rng(0)))
    inp = x[:, :-1]
    with torch.no_grad():
        lg = m(inp).reshape(-1, V.VOCAB_SIZE)
    l1 = F.cross_entropy(lg, x[:, 1:].reshape(-1), ignore_index=V.PAD)
    l2 = F.cross_entropy(lg, T.mask_bad_bricks(x)[:, 1:].reshape(-1), ignore_index=V.PAD)
    assert torch.equal(l1, l2)


def test_d4_R_token_matched_within_1pct():
    tok = lambda a: sum(map(len, _d4_seqs(a)))
    r, rf = tok("R"), tok("RF")
    assert abs(r - rf) / rf < 0.01, (r, rf)
    assert tok("RFs") == rf
