"""LoRA SFT on (caption -> bricks) chats, assistant-only loss. fp16 (T4) by default.

C1  : --n_train N   from the dataset train split (fixed seed)
C1+ : --n_train 2N  (equal-compute control is done by matching steps; see --max_steps)
C2/C3 later reuse this with --data_jsonl of {"messages": [...]} rows.

Outputs: <out>/adapter (LoRA), <out>/merged (fp16 full model for vLLM), <out>/train_log.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from . import prompts


def dataset_messages(n: int, seed: int, cache_dir=None, max_bricks: int | None = None) -> list[list[dict]]:
    from .data import load_split, DEFAULT_CACHE
    recs = load_split("train", cache_dir or DEFAULT_CACHE)
    if max_bricks:
        recs = [r for r in recs if r["n_bricks"] <= max_bricks]
    rng = random.Random(seed)
    recs = sorted(recs, key=lambda r: r["structure_id"])
    rng.shuffle(recs)
    out = []
    for r in recs[:n]:
        cap = rng.choice(r["captions"])
        out.append(prompts.initial_messages(cap) + [{"role": "assistant", "content": r["bricks"].strip()}])
    return out


def tokenize(tok, msgs: list[dict], max_len: int):
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    if not full.startswith(prompt):  # some templates differ; fall back to concatenation
        full = prompt + msgs[-1]["content"] + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels[:len(f_ids)]}


def chunked_ce(hidden, weight, labels, chunk: int):
    """Causal-LM CE without materialising full-vocab logits: per-chunk logits, recomputed in backward."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    h = hidden[:, :-1].reshape(-1, hidden.size(-1))
    y = labels[:, 1:].reshape(-1)
    keep = y != -100
    h, y = h[keep], y[keep]
    if y.numel() == 0:
        return hidden.sum() * 0

    def piece(hc, yc):
        # under fp16 AMP hidden states can be fp32 while lm_head is fp16: match the weight dtype
        return F.cross_entropy((hc.to(weight.dtype) @ weight.t()).float(), yc, reduction="sum")

    total = sum(checkpoint(piece, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
                for i in range(0, y.numel(), chunk))
    return total / y.numel()


def _trainer_cls():
    from transformers import Trainer

    class ChunkedCETrainer(Trainer):
        def __init__(self, *args, ce_chunk: int = 512, **kw):
            super().__init__(*args, **kw)
            self.ce_chunk = ce_chunk
            # our loss is already a per-batch mean -> let Trainer divide by grad_accum itself
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs, output_hidden_states=True, logits_to_keep=1)
            w = model.get_output_embeddings().weight
            loss = chunked_ce(out.hidden_states[-1], w, labels, self.ce_chunk)
            return (loss, out) if return_outputs else loss

    return ChunkedCETrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--data_jsonl", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--probe_steps", type=int, default=0, help="run N steps, print peak memory, exit (no save)")
    ap.add_argument("--ce_chunk", type=int, default=512, help="tokens per chunk for the memory-light CE loss")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=3840)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--no_merge", action="store_true")
    ap.add_argument("--save_steps", type=int, default=0, help=">0: checkpoint every N steps into <out>/ckpt and resume from the latest")
    ap.add_argument("--cpu", action="store_true", help="local dry run: fp32 on CPU")
    ap.add_argument("--stop_at", type=int, default=0,
                    help=">0: keep the LR schedule of --max_steps but stop (and checkpoint) at this step; "
                         "rerun with a larger value to continue (needs --save_steps)")
    ap.add_argument("--eval_jsonl", default=None,
                    help="held-out {messages} rows: validation loss every --eval_steps (same tokenisation/length cap)")
    ap.add_argument("--eval_steps", type=int, default=40)
    ap.add_argument("--snap_steps", type=int, default=0,
                    help=">0: keep a copy of the LoRA adapter every N steps in <out>/traj/step_N (needs --save_steps dividing N)")
    a = ap.parse_args()

    import torch
    ChunkedCETrainer = _trainer_cls()
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.data_jsonl:
        msgs = [json.loads(l)["messages"] for l in open(a.data_jsonl, encoding="utf-8") if l.strip()]
    else:
        msgs = dataset_messages(a.n_train, a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows_all = [tokenize(tok, m, 10 ** 9) for m in msgs]
    dropped_idx = [i for i, r in enumerate(rows_all) if len(r["input_ids"]) > a.max_len]
    rows = [r for r in rows_all if len(r["input_ids"]) <= a.max_len]
    n_dropped = len(rows_all) - len(rows)
    print(f"dropped {n_dropped}/{len(rows_all)} examples longer than {a.max_len} tok", flush=True)
    print(f"{len(rows)} examples, mean len {sum(len(r['input_ids']) for r in rows) / len(rows):.0f} tok", flush=True)
    eval_rows = None
    if a.eval_jsonl:
        em = [json.loads(l)["messages"] for l in open(a.eval_jsonl, encoding="utf-8") if l.strip()]
        eval_rows = [r for r in (tokenize(tok, m, 10 ** 9) for m in em) if len(r["input_ids"]) <= a.max_len]
        print(f"{len(eval_rows)} held-out examples for validation loss every {a.eval_steps} steps", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32 if a.cpu else torch.float16, device_map=None if a.cpu else {"": 0})
    except (ValueError, KeyError):  # VLM checkpoints (Qwen3.5) may need the image-text class
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.float32 if a.cpu else torch.float16, device_map=None if a.cpu else {"": 0})
    print("loaded", type(model).__name__, flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    # fp16 AMP needs fp32 trainable params
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    import inspect
    targs = dict(output_dir=str(out / "ckpt"), per_device_train_batch_size=a.bs,
                 gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs,
                 max_steps=a.probe_steps or a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine",
                 logging_steps=a.logging_steps,
                 save_strategy="steps" if a.save_steps and not a.probe_steps else "no", save_steps=a.save_steps or 500, save_total_limit=1, fp16=not a.cpu, use_cpu=a.cpu,
                 group_by_length=True, report_to=[], seed=a.seed, dataloader_num_workers=2)
    if eval_rows and not a.probe_steps:
        targs.update(eval_strategy="steps", evaluation_strategy="steps", eval_steps=a.eval_steps,
                     per_device_eval_batch_size=a.bs, prediction_loss_only=True)
    n_steps = a.max_steps if a.max_steps > 0 else int(len(rows) * a.epochs / (a.bs * a.grad_accum))
    targs["warmup_steps"] = max(1, int(0.03 * n_steps))  # warmup_ratio was removed in transformers 5
    try:
        if a.cpu:
            raise ImportError("cpu dry run")
        import bitsandbytes  # noqa: F401
        targs["optim"] = "paged_adamw_8bit"
    except Exception as e:
        print("bitsandbytes unavailable, using adamw_torch:", repr(e)[:100], flush=True)
    ok = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = [k for k in targs if k not in ok]
    if dropped:
        print("TrainingArguments: dropping unsupported", dropped, flush=True)
    args = TrainingArguments(**{k: v for k, v in targs.items() if k in ok})
    tr = ChunkedCETrainer(ce_chunk=a.ce_chunk, model=model, args=args, train_dataset=rows,
                 eval_dataset=eval_rows if (eval_rows and not a.probe_steps) else None,
                 data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    from transformers import TrainerCallback
    import uuid
    session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_id = out.name

    def event(kind, step, **kw):
        """One line of <out>/trajectory.jsonl, flushed + fsynced immediately. Written to the local disk only:
        on Kaggle it reaches persistent storage when the session ends normally (time guard), not on a hard kill."""
        if a.probe_steps:
            return
        with open(out / "trajectory.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": kind, "run_id": run_id, "session_id": session_id, "step": step,
                                "time": time.time(), **kw}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # stop_at milestones and the final step get a validation loss and a kept snapshot at the same step
    aligned = {s for s in (a.stop_at, a.max_steps) if s and s > 0}

    class _Trajectory(TrainerCallback):
        """Live trajectory (train / validation / snapshot / resume / end events); resumed runs append, and the
        resume event's resume_from_step tells the plotter which earlier lines belong to a rolled-back branch."""
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                kind = "validation" if "eval_loss" in logs else ("train" if "loss" in logs else "summary")
                event(kind, state.global_step, **{"epoch": state.epoch, **logs})

        def on_step_end(self, args, state, control, **kw):
            if state.global_step in aligned and eval_rows:
                control.should_evaluate = True
            return control

        def on_save(self, args, state, control, **kw):
            if a.snap_steps and (state.global_step % a.snap_steps == 0 or state.global_step in aligned):
                import shutil
                ck = out / "ckpt" / f"checkpoint-{state.global_step}"
                dst = out / "traj" / f"step_{state.global_step}"
                dst.mkdir(parents=True, exist_ok=True)
                for fn in ("adapter_model.safetensors", "adapter_config.json"):
                    if (ck / fn).exists():
                        shutil.copy(ck / fn, dst / fn)
                event("snapshot", state.global_step, epoch=state.epoch, checkpoint_id=f"{run_id}/step_{state.global_step}",
                      path=str(dst.relative_to(out)))
    tr.add_callback(_Trajectory())
    if a.stop_at:
        assert a.save_steps and a.max_steps > 0 and a.stop_at <= a.max_steps, "--stop_at needs --save_steps and --max_steps"
        class _StopAt(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):
                if state.global_step >= a.stop_at:
                    control.should_save = True
                    control.should_training_stop = True
                return control
        tr.add_callback(_StopAt())
    if torch.cuda.is_available() and not a.cpu:
        torch.cuda.reset_peak_memory_stats()
    ck = sorted((out / "ckpt").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if a.save_steps and not a.probe_steps else []  # probes never save/resume
    if ck:
        print("resuming from", ck[-1], flush=True)
    resume_step = int(ck[-1].name.split("-")[1]) if ck else 0
    if not a.probe_steps:
        import hashlib, platform, subprocess
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                    cwd=Path(__file__).resolve().parent).stdout.strip()
        except Exception:
            commit = ""
        data_path = a.data_jsonl
        man = {"run_id": run_id, "args": vars(a), "git_commit": commit, "base_model": a.model,
               "data_sha256": hashlib.sha256(open(data_path, "rb").read()).hexdigest() if data_path else None,
               "eval_sha256": hashlib.sha256(open(a.eval_jsonl, "rb").read()).hexdigest() if a.eval_jsonl else None,
               "n_examples": len(rows), "n_dropped_too_long": n_dropped, "dropped_row_indices": dropped_idx,
               "n_eval_examples": len(eval_rows) if eval_rows else 0,
               "effective_batch": a.bs * a.grad_accum, "planned_steps": n_steps,
               "loss": "mean NLL over assistant tokens per micro-batch (prompt/padding masked, long rows dropped "
                       "not truncated); micro-batches weighted equally across gradient accumulation; eval_loss = "
                       "mean of per-batch losses over the held-out set",
               "lora": {"r": a.rank, "alpha": 2 * a.rank, "dropout": 0.05, "target_modules": "all-linear"},
               "training_args": {k: str(v) for k, v in args.to_dict().items()},
               "versions": {m: __import__(m).__version__ for m in ("torch", "transformers", "peft")},
               "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() and not a.cpu else platform.processor(),
               "python": platform.python_version()}
        mp = out / "manifest.json"
        old = json.loads(mp.read_text()) if mp.exists() else {}
        man["sessions"] = old.get("sessions", []) + [{"session_id": session_id, "resume_from_step": resume_step,
                                                      "git_commit": commit, "start": time.time()}]
        mp.write_text(json.dumps(man, indent=1))
        event("resume", resume_step, resume_from_step=resume_step, git_commit=commit)
        if eval_rows and resume_step == 0:  # step-0 baseline validation loss (the untouched base model + zero LoRA)
            tr.evaluate()
    try:
        tr.train(resume_from_checkpoint=str(ck[-1]) if ck else None)
    except BaseException as e:
        event("end", tr.state.global_step, status="failed", reason=repr(e)[:300])
        raise
    event("end", tr.state.global_step, status="stopped_at_milestone" if a.stop_at and tr.state.global_step < (a.max_steps or 0) else "completed")
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() and not a.cpu else 0.0
    print(f"PEAK_MEM_GB {peak:.2f}", flush=True)
    if a.probe_steps:
        print(f"probe done: {a.probe_steps} steps in {(time.time() - t0) / 60:.1f} min", flush=True)
        return
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    log = {"args": vars(a), "n_examples": len(rows), "n_dropped_too_long": n_dropped, "peak_mem_gb": peak, "train_seconds": time.time() - t0,
           "log_history": tr.state.log_history}
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    if not a.no_merge:
        merged = model.merge_and_unload()
        merged = merged.half()
        merged.save_pretrained(out / "merged", safe_serialization=True)
        tok.save_pretrained(out / "merged")
        try:  # VLM checkpoints need processor/preprocessor configs for vLLM
            from transformers import AutoProcessor
            AutoProcessor.from_pretrained(a.model).save_pretrained(out / "merged")
        except Exception as e:
            print("no processor saved:", repr(e)[:120])
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
