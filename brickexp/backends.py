"""Pluggable generation backends: generate(conversations, metas) -> one string per conversation."""
from __future__ import annotations

import re


class Backend:
    name = "base"

    def generate(self, conversations: list[list[dict]], metas: list[dict]) -> list[str]:
        raise NotImplementedError


class ReplayBackend(Backend):
    """Returns the dataset's ground-truth bricks (local pipeline test).

    fail_first: first N attempts per sample return a deliberately floating brick (exercises retries).
    """
    name = "replay"

    def __init__(self, fail_first: int = 0):
        self.fail_first = fail_first

    def generate(self, conversations, metas):
        return ["2x4 (0,0,5)\n" if m["attempt"] < self.fail_first else m["bricks"] for m in metas]


class VLLMBackend(Backend):
    """vLLM chat generation (lazy import so CPU-only machines can import this module)."""
    name = "vllm"

    def __init__(self, model: str, temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 4096,
                 dtype: str = "half", tensor_parallel_size: int = 1, max_model_len: int = 16384,
                 gpu_memory_utilization: float = 0.9, seed: int = 0, enable_thinking: bool = False,
                 **llm_kwargs):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, dtype=dtype, tensor_parallel_size=tensor_parallel_size,
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       seed=seed, **llm_kwargs)
        self.params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.enable_thinking = enable_thinking
        self.model_name = model

    def generate(self, conversations, metas):
        outs = self.llm.chat(conversations, self.params, use_tqdm=False,
                             chat_template_kwargs={"enable_thinking": self.enable_thinking})
        return [strip_think(o.outputs[0].text) for o in outs]


class HFBackend(Backend):
    """transformers.generate fallback (slower; use if vLLM can't run the model on the GPU)."""
    name = "hf"

    def __init__(self, model: str, temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 4096,
                 batch_size: int = 8, enable_thinking: bool = False, adapter_base: str | None = None, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if adapter_base:  # e.g. BrickGPT = LoRA on gated Llama-3.2-1B-Instruct; load an ungated mirror of the base
            from peft import PeftModel
            self.tok = AutoTokenizer.from_pretrained(adapter_base, padding_side="left")
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(adapter_base, dtype=torch.float16, device_map="auto")
            self.model = PeftModel.from_pretrained(base, model).merge_and_unload().eval()
            self.kw = dict(do_sample=temperature > 0, temperature=temperature, top_p=top_p, max_new_tokens=max_tokens)
            self.bs, self.enable_thinking, self.model_name = batch_size, enable_thinking, f"{model}@{adapter_base}"
            return
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16, device_map="auto")
        except (ValueError, KeyError):  # VLM checkpoints (e.g. Qwen3.5) may need the image-text class
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.float16, device_map="auto")
        self.model.eval()
        self.kw = dict(do_sample=temperature > 0, temperature=temperature, top_p=top_p, max_new_tokens=max_tokens)
        self.bs, self.enable_thinking, self.model_name = batch_size, enable_thinking, model

    def generate(self, conversations, metas):
        import torch
        outs = []
        for i in range(0, len(conversations), self.bs):
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=self.enable_thinking)
                     for c in conversations[i:i + self.bs]]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, **self.kw, pad_token_id=self.tok.pad_token_id)
            outs += [strip_think(t) for t in
                     self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        return outs


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.split("</think>")[-1].strip()


def make_backend(kind: str, **kw) -> Backend:
    if kind == "replay":
        return ReplayBackend(**kw)
    if kind == "vllm":
        return VLLMBackend(**kw)
    if kind == "hf":
        return HFBackend(**kw)
    raise ValueError(kind)
