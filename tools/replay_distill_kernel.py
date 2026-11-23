"""Local replay of notebooks/kaggle_distill_sft.ipynb before any Kaggle push (debug-before-resubmit).

What is real: bundle build (tools/make_kaggle_bundle.py), Kaggle-style extraction, every notebook cell in order,
brickexp.train_sft (LoRA, --cpu), brickexp.eval_2x2 run/table, output file layout.
What is substituted (no GPU / no vLLM / no model download locally):
  - pip cell skipped (deps come from the replay interpreter)
  - model = a tiny randomly initialised Qwen2 + a BPE tokenizer trained on the SFT data (built here, no download)
  - eval backend vllm -> replay (scripted outputs; exercises the eval loop, checker and table code)
  - Kaggle Secrets absent -> notebook falls back as written; checker forced to 'auto' (Gurobi self-check not required)
python tools/replay_distill_kernel.py --tmp <dir>
"""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tiny_model(d, data_jsonl):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    texts = []
    for l in open(data_jsonl, encoding="utf-8"):
        for m in json.loads(l)["messages"]:
            texts.append(m["content"])
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
    sp = ["<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=sp,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    ft = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<|im_end|>", pad_token="<|endoftext|>")
    ft.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    ft.save_pretrained(d)
    cfg = Qwen2Config(vocab_size=len(ft), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=True,
                      eos_token_id=ft.eos_token_id, pad_token_id=ft.pad_token_id)
    Qwen2ForCausalLM(cfg).save_pretrained(d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tmp", required=True)
    ap.add_argument("--nb", default="notebooks/kaggle_distill_sft.ipynb")
    ap.add_argument("--runs", type=int, default=1, help="colab: run twice to test skip-on-rerun")
    ap.add_argument("--need", nargs="*", default=["TABLE_2x2.md", "sft/train_log.json"])
    ap.add_argument("--adapt_root", default=None, help="local dir standing in for /kaggle/input adapters")
    a = ap.parse_args()
    t = Path(a.tmp); shutil.rmtree(t, ignore_errors=True)
    inp = t / "input/brickexp-code"; inp.mkdir(parents=True); work = t / "working"; work.mkdir()
    subprocess.run([sys.executable, str(ROOT / "tools/make_kaggle_bundle.py")], check=True)
    shutil.copy(ROOT / "dist/brickexp_code.zip", inp / "brickexp_code.zip")
    with zipfile.ZipFile(inp / "brickexp_code.zip") as z:
        names = z.namelist()
    assert not any(b in n for n in names for b in ("gurobi.lic", ".env", "deepseek")), "secret-looking file in bundle"
    assert "data/distill/sft_distill.jsonl" in names, "SFT data missing from bundle"
    tm = t / "tiny_qwen"; tiny_model(tm, ROOT / "data/distill/sft_distill.jsonl")
    nb = json.loads((ROOT / a.nb).read_text(encoding="utf-8"))
    shutil.copy(inp / "brickexp_code.zip", work / "brickexp_code.zip")  # colab: zip sits in the Drive OUTD
    cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    src = []
    for c in cells:
        if 'stage("pip")' in c:  # keep helpers, drop the pip install part
            c = c.split('stage("pip")')[0]
        c = c.replace('"/kaggle/working" if os.path.exists("/kaggle") else "/content/work"', repr(work.as_posix()))
        c = c.replace('from google.colab import drive\ndrive.mount("/content/drive")\n', '')
        c = c.replace('"/content/drive/MyDrive/brickexp_distill"', repr(work.as_posix()))
        c = c.replace('"/content/drive/MyDrive/brickexp_mix"', repr(work.as_posix()))
        c = c.replace('OUTD = "/kaggle/working"', "OUTD = " + repr(work.as_posix()))
        if a.adapt_root:
            c = c.replace('ADAPT_ROOT = "/kaggle/input"', "ADAPT_ROOT = " + repr(Path(a.adapt_root).as_posix()))
        c = c.replace("--data data/mix/heldout200.jsonl", "--data data/mix/heldout200.jsonl --cpu --max_len 100000")
        c = c.replace('"/kaggle/temp/RLtheLEGO"', repr((t / "code").as_posix()))
        c = c.replace("math.ceil(nB / 16)", "3").replace("--save_steps 40", "--save_steps 1")
        c = c.replace('glob.glob("/content/**/brickexp_code.zip", recursive=True)', '[]')
        c = c.replace('"/content/RLtheLEGO"', repr((t / "code").as_posix()))
        c = c.replace('"/kaggle/input/**/brickexp_code.zip"', repr((t / "input").as_posix() + "/**/brickexp_code.zip"))
        c = c.replace('if "gurobi" == "gurobi" and not OK', 'if False')
        c = c.replace("Qwen/Qwen2.5-1.5B-Instruct", tm.as_posix())
        c = c.replace("--backend vllm", "--backend replay").replace("--checker gurobi", "--checker auto")
        c = c.replace("--limit 0", "--limit 1").replace("--max_steps -1", "--max_steps 2 --cpu").replace("--max_steps {STEPS}", "--max_steps {STEPS} --cpu")
        c = c.replace("--max_len 3072", "--max_len 1024")
        c = c.replace('assert PEAK is not None and PEAK < 13.5', 'assert PEAK is None or PEAK < 13.5')
        c = c.replace("export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ", "")
        src.append(c)
    BASH = str(Path(shutil.which("git")).parent.parent / "bin" / "bash.exe")
    pre = ("import subprocess as _sp\n_run = _sp.run\n"
           "def _r(cmd, *a, executable=None, shell=False, **k):\n"
           f"    if executable == '/bin/bash': return _run([{BASH!r}, '-c', cmd], *a, **k)\n"
           "    return _run(cmd, *a, shell=shell, executable=executable, **k)\n_sp.run = _r\n")
    code = pre + "\n\n".join(src).replace("python -m brickexp", f"{Path(sys.executable).as_posix()} -m brickexp")
    (t / "kernel.py").write_text(code, encoding="utf-8")
    import os
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")  # Kaggle is Linux/UTF-8; Windows default is GBK
    for i in range(a.runs):
        r = subprocess.run([sys.executable, str(t / "kernel.py")], cwd=t, env=env)
        print(f"=== replay pass {i + 1}: rc {r.returncode}", flush=True)
        if r.returncode:
            break
    outs = sorted(p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file() and "RLtheLEGO" not in p.parts and "merged" not in p.parts)
    print("replay rc", r.returncode, "| outputs:", outs[:40])
    need = a.need
    missing = [n for n in need if not (work / n).exists()]
    print("missing:", missing, "| error.txt:", (work / "error.txt").exists())
    sys.exit(r.returncode or bool(missing))


if __name__ == "__main__":
    main()
