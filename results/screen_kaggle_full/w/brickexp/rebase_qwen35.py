"""Turn a merged text-only Qwen3.5 checkpoint (Qwen3_5ForCausalLM, saved by transformers 5) into the base
repo's layout (Qwen3_5ForConditionalGeneration) so vLLM 0.17.1 and transformers 4.57 can load it.

Why: vLLM 0.17.1 registers only Qwen3_5ForConditionalGeneration, and the merged tokenizer_config names
Qwen3_5Tokenizer, which transformers 4.57.6 lacks (DEBUG_LOG 2026-11-18). The merged weights already use the base
key names (model.language_model.*), so we copy the base repo (config, tokenizer, vision + MTP weights) and swap in
every merged tensor, cast to the base dtype.

Usage: python -m brickexp.rebase_qwen35 MERGED_DIR OUT_DIR [--base Qwen/Qwen3.5-2B]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def rebase(merged_dir, out_dir, base="Qwen/Qwen3.5-2B"):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_dir = Path(snapshot_download(base))
    merged_dir, out_dir = Path(merged_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import load_file
    merged = {}
    for f in sorted(merged_dir.glob("*.safetensors")):  # read once into RAM (per-tensor reopen took 67 min on /kaggle/input)
        merged.update(load_file(str(f)))
    used = set()
    for f in base_dir.iterdir():
        if f.suffix == ".safetensors":
            tensors = {}
            with safe_open(str(f), "pt") as h:
                meta = h.metadata()
                for k in h.keys():
                    t = h.get_tensor(k)
                    if k in merged:
                        m = merged[k]
                        assert m.shape == t.shape, (k, m.shape, t.shape)
                        t = m.to(t.dtype)
                        used.add(k)
                    tensors[k] = t.contiguous()
            save_file(tensors, str(out_dir / f.name), metadata=meta or {"format": "pt"})
        elif f.is_file():
            shutil.copy(f, out_dir / f.name)
    missing = sorted(set(merged) - used)
    # tied embeddings: lm_head may be absent from the base; anything else unmatched is an error
    missing = [k for k in missing if k != "lm_head.weight"]
    assert not missing, f"merged tensors not in base: {missing[:5]} (+{len(missing) - 5})"
    info = {"base": base, "base_dir": str(base_dir), "replaced": len(used), "merged_total": len(merged)}
    (out_dir / "rebase_info.json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("out")
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    a = ap.parse_args()
    print(json.dumps(rebase(a.merged, a.out, a.base)))
