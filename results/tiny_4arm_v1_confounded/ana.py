import json, sys, collections, numpy as np
p = sys.argv[1]
S = [json.loads(l) for l in open(p)]
c0 = collections.Counter(s["attempts"][0]["failure"] for s in S)
call = collections.Counter(a["failure"] for s in S for a in s["attempts"])
print("n", len(S), "k0 fail", dict(c0))
print("all attempts fail", dict(call))
nb = [s["attempts"][0]["n_bricks"] for s in S]
print("k0 bricks mean/med/max", np.mean(nb), np.median(nb), max(nb), "zero:", sum(x == 0 for x in nb))
print("bad_tokens mean", np.mean([s["attempts"][0].get("n_bad_tokens", 0) for s in S]),
      "n_back mean", np.mean([s["attempts"][0].get("n_back", 0) for s in S]))
print("solved_at", collections.Counter(s["solved_at"] for s in S))
for s in S[:3]:
    a = s["attempts"][0]
    print("--- cat", s["cat"], a["failure"], a["n_bricks"], "\n", a["text"][:400])
