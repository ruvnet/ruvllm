"""Cloud workload. It receives only original synthetic prompts, never tests or credentials.

It generates text only. Execution and scoring happen in the local sandbox.
"""
import base64
import gzip
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
REVISION = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"


def emit(kind, data):
    print("RUFLO_RESULT " + json.dumps({"kind": kind, "data": data}), flush=True)


def main():
    torch.manual_seed(20261102)
    torch.set_num_threads(8)
    cases = json.loads(gzip.decompress(base64.b64decode(os.environ["RUFLO_CASES_GZIP"])))
    emit("hardware", {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                      "model": MODEL, "revision": REVISION, "max_tokens": 768, "temperature": 0})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False, token=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False,
        token=False, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.eval()
    for case in cases:
        try:
            started = time.monotonic()
            prompt = tokenizer.apply_chat_template(case["messages"], tools=case["tools"] or None,
                add_generation_prompt=True, tokenize=False, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=768, do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0, inputs["input_ids"].shape[1]:].tolist()
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            emit("response", {"id": case["id"], "text": tokenizer.decode(generated, skip_special_tokens=False),
                "input_tokens": inputs["input_ids"].shape[1], "output_tokens": len(generated),
                "seconds": round(time.monotonic() - started, 3),
                "finish_reason": "stop" if generated and generated[-1] in eos else "length"})
        except Exception as e:
            emit("response", {"id": case["id"], "text": "", "input_tokens": 0, "output_tokens": 0,
                "seconds": 0, "finish_reason": "error", "error": type(e).__name__})
    emit("completed", {"cases": len(cases)})


if __name__ == "__main__":
    main()
