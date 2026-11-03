"""Build original math fixtures and freeze all comparisons before inference."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import random
import shutil
import urllib.request
from pathlib import Path
from common import HERE, ROOT, jsonl, read, sha, write

SEED = 2026110307
MODELS = [
    ("qwen035-08b-base", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    ("qwen035-4b-base", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ("qwen3-coder-30b", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"),
]


def math_cases():
    rng = random.Random(SEED)
    result = []
    for family in range(16):
        for i in range(8):
            a, b, c = [rng.randint(11, 49) for _ in range(3)]
            n = rng.randint(6, 14)
            if family == 4: n = 9+i
            if family == 0:
                q, answer = f"Compute {a} times {b}, then subtract {c}.", a*b-c
            elif family == 1:
                q, answer = f"Find the greatest common divisor of {a*n} and {b*n}.", math.gcd(a*n,b*n)
            elif family == 2:
                q, answer = f"Find the least common multiple of {a} and {b}.", math.lcm(a,b)
            elif family == 3:
                q, answer = f"A sequence starts at {a} and increases by {b} each term. What is its {n}th term?", a+(n-1)*b
            elif family == 4:
                q, answer = f"How many unordered groups of three distinct people can be selected from {n} people?", math.comb(n,3)
            elif family == 5:
                q, answer = f"What is the remainder when {a} to the power {n} is divided by {b}?", pow(a,n,b)
            elif family == 6:
                q, answer = f"A rectangle has perimeter {2*(a+b)} and width {a}. What is its area?", a*b
            elif family == 7:
                q, answer = f"An integer x satisfies {a}x + {b} = {a*c+b}. What is x?", c
            elif family == 8:
                q, answer = f"A box contains only red and blue tokens in ratio {a} to {b}. There are {(a+b)*n} tokens. How many are blue?", b*n
            elif family == 9:
                q, answer = f"A price of {100*a} dollars is reduced by {b} percent. What is the new price in dollars?", a*(100-b)
            elif family == 10:
                q, answer = f"Compute the sum of all integers from {a} through {a+n} inclusive.", (n+1)*(2*a+n)//2
            elif family == 11:
                q, answer = f"For f(x) = {a}x squared minus {b}x plus {c}, compute f({n}).", a*n*n-b*n+c
            elif family == 12:
                q, answer = f"Two integers have sum {a+b} and difference {abs(a-b)}. What is the larger integer?", max(a,b)
            elif family == 13:
                q, answer = f"There are {a} crates containing {b} items each. After selling {c} items, how many remain?", a*b-c
            elif family == 14:
                q, answer = f"A right triangle has perpendicular side lengths {2*a} and {b}. What is its area?", a*b
            else:
                q, answer = f"What is the largest integer k such that {b}k is at most {a*b+c}?", (a*b+c)//b
            question = q + " Explain your solution and put the final integer with no units in \\boxed{}."
            result.append({"id": f"fresh-math-{family:02d}-{i:02d}", "kind": "math", "panel": "math128",
                           "family": f"procedural-{family:02d}", "answer": answer,
                           "parameters": [a,b,c,n], "messages": [{"role":"user","content": question}], "tools": []})
    assert len(result) == len({r["messages"][0]["content"] for r in result}) == 128
    old = read(ROOT / "research/weight_evolution/cases.json")
    assert not {r["messages"][0]["content"] for r in result} & {r["question"] for r in old}
    return result


def prepare(work):
    if (HERE / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen")
    work = Path(work)
    inputs = work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    cases = [{**c, "panel": "humaneval164"} for c in jsonl(ROOT / "data/external/humaneval.jsonl")]
    cases += [{**c, "panel": "tools27"} for c in jsonl(ROOT / "data/release.jsonl") if c["kind"] == "tool"]
    cases += math_cases()
    assert len(cases) == len({c["id"] for c in cases}) == 319
    prompts = [{k:v for k,v in c.items() if k in ("id","kind","panel","family","messages","tools")} for c in cases]
    write(HERE / "cases.json", cases)
    write(HERE / "prompts.json", prompts)
    pins = []
    for key, model, revision in MODELS:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true") as response:
            meta = json.load(response)
        assert meta["sha"] == revision
        weights = {s["rfilename"]: s["lfs"]["sha256"] for s in meta["siblings"] if s["rfilename"].endswith(".safetensors")}
        pins.append({"key":key,"model":model,"revision":revision,"license":"Apache-2.0","weights_sha256":weights})
    write(HERE / "models.json", pins)
    source_inputs = {
        "patch.json": ROOT / "runtime/weight-evolution/alpha-100/patch.json",
        "policy.json": ROOT / "runtime/weight-evolution/alpha-100/policy.json",
        "adapter_model.safetensors": ROOT / "runtime/cloud-train/download/step-064/adapter_model.safetensors",
        "adapter_config.json": ROOT / "runtime/cloud-train/download/step-064/adapter_config.json",
    }
    for name, path in source_inputs.items():
        dest=inputs/name
        if dest.exists() and sha(dest) != sha(path):
            raise ValueError("Conflicting input bundle")
        if not dest.exists(): shutil.copyfile(path,dest)
    write(HERE / "assets.json", {name:sha(path) for name,path in source_inputs.items()})
    print(json.dumps({"cases":len(cases),"assets":str(inputs)}))


def freeze():
    variants = [k for k,_,_ in MODELS]
    variants.insert(1,"qwen035-08b-edited")
    variants.insert(3,"qwen035-4b-lora")
    arms = [{"id":v+"-direct","variant":v,"thinking":False,"panels":["humaneval164","tools27","math128"]} for v in variants]
    arms += [{"id":v+"-thinking","variant":v,"thinking":True,"panels":["math128"]} for v in variants[:2]]
    implementation = [p.name for p in HERE.glob("*.py")]+["../../ruflo_model/scoring.py"]
    protocol = {
        "schema":"ruflo.vast-cross-model.v1", "created_at":datetime.now(timezone.utc).isoformat(),
        "purpose":"frozen diagnostic comparison; no selection, training, promotion, or SOTA claim",
        "arms":arms, "seed":SEED, "temperature":0, "top_p":1, "top_k":-1, "repetition_penalty":1,
        "panel_max_tokens":{"humaneval164":2048,"tools27":2048,"math128":8192},
        "total_generations":1851,"maximum_output_tokens":9295872,"maximum_wall_seconds":20700,
        "container_image":"docker.io/vllm/vllm-openai@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b",
        "runtime_versions":{"vllm":"0.21.0","transformers":"5.12.1","torch":"2.11.0"},
        "eos_token_ids":{"qwen035-08b-base":[248044],"qwen035-08b-edited":[248044],
                         "qwen035-4b-base":[248044],"qwen035-4b-lora":[248044],
                         "qwen3-coder-30b":[151645,151643]},
        "engine":{"dtype":"bfloat16","tensor_parallel_size":1,"max_model_len":12288,
                  "max_num_seqs":8,"max_num_batched_tokens":8192,"gpu_memory_utilization":0.90,
                  "enforce_eager":True,"enable_prefix_caching":False,"limit_mm_per_prompt":{"image":0,"video":0}},
        "batch_size":8,"precision":"BF16 all variants; FP32 LoRA merge then BF16 rounding",
        "input_sha256":{n:sha(HERE/n) for n in ["cases.json","prompts.json","models.json","assets.json"]},
        "implementation_sha256":{n:sha(HERE/n) for n in sorted(implementation)},
        "comparisons":[["qwen035-08b-base-direct","qwen035-08b-edited-direct"],
                       ["qwen035-08b-base-thinking","qwen035-08b-edited-thinking"],
                       ["qwen035-4b-base-direct","qwen035-4b-lora-direct"]],
        "reference":"qwen3-coder-30b-direct is a separate larger model, never a patch target",
        "metrics":["answer_or_test_correct","completed","delivered_correct","output_tokens","batch_wall_seconds"],
        "strongest_unedited_profile":"thinking disabled from prior dev; all 0.8B math profiles disclosed",
        "scoring":"local frozen strict tool schema and macOS sandbox; integer exact boxed answer; no model judge",
        "limitations":["HumanEval pretraining contamination unknown; repeated public benchmark",
                      "27 tool fixtures are synthetic and too small for production qualification",
                      "math corpus is original procedural arithmetic, not MATH500 or advanced reasoning",
                      "greedy fixed chat protocol is not an official leaderboard score",
                      "one GPU and seed; batch throughput is not individual request latency",
                      "prior local 16 case weight repair gate failed; cloud run is diagnostic replication"],
    }
    write(HERE / "protocol.json",protocol)
    write(HERE / "freeze.json",{"protocol_sha256":sha(HERE/"protocol.json"),"selection_use":False})
    print(json.dumps({"protocol_sha256":sha(HERE/"protocol.json"),"generations":1851}))


if __name__ == "__main__":
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare","freeze"])
    p.add_argument("--work",default=str(ROOT/"runtime/vast-cross-model"))
    args=p.parse_args()
    prepare(args.work) if args.action=="prepare" else freeze()
