"""Run a Hugging Face model (optionally with a LoRA adapter) as the agent, locally.

    python evals/agent/run_eval.py --model Qwen/Qwen3-1.7B --backend hf
    python evals/agent/run_eval.py --model Qwen/Qwen3-1.7B --backend hf --adapter training/agent/out/lora

Returns messages in the same shape as the OpenRouter API, so run_eval's loop
is unchanged. Tool calls are parsed from the Hermes/Qwen format:
<tool_call>{"name": ..., "arguments": {...}}</tool_call>
"""

from __future__ import annotations

import json
import re

_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


class LocalModel:
    def __init__(self, name: str, adapter: str | None = None, four_bit: bool = True, max_new_tokens: int = 768):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tok = AutoTokenizer.from_pretrained(name)
        kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
        if four_bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.calls = 0

    def prompt(self, messages, tools) -> str:
        return self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)

    def chat(self, messages, tools):
        import torch

        text = self.prompt(messages, tools)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        reply = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"choices": [{"message": parse(reply, self.calls_id())}]}

    def calls_id(self):
        self.calls += 1
        return self.calls


def parse(reply: str, n: int) -> dict:
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    calls = []
    for i, raw in enumerate(_CALL.findall(reply)):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", {})
        calls.append({"id": f"l{n}_{i}", "type": "function",
                      "function": {"name": obj.get("name", ""), "arguments": args if isinstance(args, str) else json.dumps(args)}})
    content = _CALL.sub("", reply).strip()
    msg = {"content": content}
    if calls:
        msg["tool_calls"] = calls
    return msg
