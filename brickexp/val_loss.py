"""Held-out validation loss of LoRA adapters (assistant tokens only, same tokenisation/length cap as train_sft).
python -m brickexp.val_loss --model Qwen/Qwen2.5-1.5B-Instruct --data data/mix/heldout200.jsonl \
       --adapters NAME=DIR ... --out val_loss.json   (base model row is always included)
"""
import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--adapters", nargs="*", default=[]); ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=3840); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from .train_sft import tokenize
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [tokenize(tok, json.loads(l)["messages"], 10 ** 9) for l in open(a.data, encoding="utf-8")]
    rows = [r for r in rows if len(r["input_ids"]) <= a.max_len]
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32 if dev == "cpu" else torch.float16).to(dev)
    base.eval()

    def loss_of(m):
        tot, n = 0.0, 0
        with torch.no_grad():
            for r in rows:
                ids = torch.tensor([r["input_ids"]], device=dev); lab = torch.tensor([r["labels"]], device=dev)
                logits = m(input_ids=ids).logits[0, :-1].float(); y = lab[0, 1:]
                k = y != -100
                tot += torch.nn.functional.cross_entropy(logits[k], y[k], reduction="sum").item(); n += int(k.sum())
        return tot / max(n, 1), n

    res = {"n_examples": len(rows), "base": None, "adapters": {}}
    t = time.time(); l, n = loss_of(base); res["base"] = {"loss": l, "tokens": n}
    print("base", l, f"{time.time() - t:.0f}s", flush=True)
    for spec in a.adapters:
        name, d = spec.split("=", 1)
        t = time.time()
        m = PeftModel.from_pretrained(base, d); m.eval()
        l, n = loss_of(m); res["adapters"][name] = {"loss": l, "tokens": n, "dir": d}
        print(name, l, f"{time.time() - t:.0f}s", flush=True)
        base = m.unload()  # drop LoRA layers before the next adapter
    open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
