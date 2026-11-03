"""Text only GPU inference. Never executes model output or receives a cloud API key."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from common import HERE, clean_environment, object_sha, read, sha, tree, verify_protocol, write


def utc():
    return datetime.now(timezone.utc).isoformat()


def runtime():
    import torch
    versions={k:importlib.metadata.version(k) for k in ("vllm","transformers","torch","safetensors","tokenizers","huggingface-hub")}
    packages=subprocess.check_output([os.sys.executable,"-m","pip","freeze"],text=True).splitlines()
    info={"versions":versions,"packages":packages,"python":platform.python_version(),"platform":platform.platform(),
          "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
          "gpu_bytes":torch.cuda.get_device_properties(0).total_memory,
          "driver":subprocess.check_output(["nvidia-smi","--query-gpu=driver_version,uuid,name,memory.total","--format=csv,noheader"],text=True).strip()}
    return info


def run(work, arm_id, deadline, preflight=False):
    clean=clean_environment()
    os.environ.clear()
    os.environ.update(clean)
    protocol, protocol_sha=verify_protocol()
    arms={a["id"]:a for a in protocol["arms"]}
    arm=arms[arm_id]
    out=work/("preflight" if preflight else "outputs")/arm_id
    out.mkdir(parents=True,exist_ok=False)
    info=runtime()
    for key,value in protocol["runtime_versions"].items():
        if info["versions"][key].split("+")[0] != value: raise ValueError(f"Runtime version mismatch: {key}")
    if info["gpu_bytes"] < 75_000_000_000: raise ValueError("Protocol requires one 80GB class GPU")
    model_path=work/"models"/arm["variant"]
    manifest=read(work/"models"/(arm["variant"]+".manifest.json"))
    observed=tree(model_path)
    if observed["sha256"] != manifest["sha256"]: raise ValueError("Model bytes differ before load")
    binding={"protocol_sha256":protocol_sha,"model_sha256":observed["sha256"],
             "model_manifest_sha256":sha(work/"models"/(arm["variant"]+".manifest.json")),
             "runtime_sha256":object_sha(info),"variant":arm["variant"],"arm":arm_id}
    write(out/"runtime.json",info)
    write(out/"binding.json",binding)
    write(out/"state.json",{"status":"loading","started_at":utc(),"deadline_epoch":deadline})
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,token=False)
    eos_ids=protocol["eos_token_ids"][arm["variant"]]
    if tokenizer.eos_token_id not in eos_ids: raise ValueError("Tokenizer EOS differs from frozen stop set")
    engine=dict(protocol["engine"])
    if arm["variant"]=="qwen3-coder-30b": engine.pop("limit_mm_per_prompt")
    started=time.monotonic()
    llm=LLM(model=str(model_path),tokenizer=str(model_path),trust_remote_code=False,hf_token=False,
            seed=protocol["seed"],generation_config="vllm",**engine)
    load_seconds=time.monotonic()-started
    if tree(model_path)["sha256"] != observed["sha256"]: raise ValueError("Model changed while loading")
    cases=[c for c in read(HERE/"prompts.json") if c["panel"] in arm["panels"]]
    if preflight:
        cases=[{"id":"runtime-smoke-unscored","kind":"math","panel":"math128","tools":[],
                "messages":[{"role":"user","content":"Reply with the integer 4 and nothing else."}]}]
    total_tokens=0
    generation_started=time.monotonic()
    rows_path=out/"responses.jsonl"
    with rows_path.open("x") as output:
        for offset in range(0,len(cases),protocol["batch_size"]):
            if time.time() >= deadline or (work/"STOP").exists(): raise TimeoutError("Absolute run deadline or STOP reached")
            batch=cases[offset:offset+protocol["batch_size"]]
            prompt_ids=[]
            parameters=[]
            for case in batch:
                prompt=tokenizer.apply_chat_template(case["messages"],tools=case["tools"] or None,
                    add_generation_prompt=True,tokenize=True,enable_thinking=arm["thinking"])
                cap=32 if preflight else protocol["panel_max_tokens"][case["panel"]]
                if len(prompt)+cap>protocol["engine"]["max_model_len"]: raise ValueError("Prompt exceeds frozen context allocation")
                prompt_ids.append(prompt)
                parameters.append(SamplingParams(temperature=0,top_p=1,top_k=-1,repetition_penalty=1,
                    max_tokens=cap,seed=protocol["seed"],skip_special_tokens=False,stop_token_ids=eos_ids))
            batch_start=time.monotonic()
            generated=llm.generate([{"prompt_token_ids":ids} for ids in prompt_ids],parameters,use_tqdm=False)
            batch_seconds=time.monotonic()-batch_start
            if len(generated)!=len(batch): raise ValueError("Missing generation")
            for case, prompt, result, sampling in zip(batch,prompt_ids,generated,parameters,strict=True):
                assert result.prompt_token_ids==prompt
                assert len(result.outputs)==1
                response=result.outputs[0]
                tokens=list(response.token_ids)
                assert tokens and response.finish_reason in ("stop","length")
                assert len(tokens)<=sampling.max_tokens
                assert response.finish_reason!="length" or len(tokens)==sampling.max_tokens
                total_tokens+=len(tokens)
                metrics=getattr(result,"metrics",None)
                metrics={key:getattr(metrics,key,None) for key in ("arrival_time","first_scheduled_time","first_token_time","last_token_time","finished_time")}
                row={**binding,"id":case["id"],"panel":case["panel"],"kind":case["kind"],
                     "thinking":arm["thinking"],"prompt_sha256":object_sha(prompt),"input_tokens":len(prompt),
                     "text":response.text,"token_ids":tokens,"output_tokens":len(tokens),
                     "max_tokens":sampling.max_tokens,"finish_reason":response.finish_reason,"stop_reason":response.stop_reason,
                     "batch_index":offset//protocol["batch_size"],"batch_seconds":batch_seconds,"metrics":metrics}
                output.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
            output.flush()
            os.fsync(output.fileno())
            write(out/"state.json",{"status":"running","heartbeat_at":utc(),"completed":offset+len(batch),
                  "output_tokens":total_tokens,"elapsed_seconds":time.monotonic()-generation_started},exclusive=False)
            print(json.dumps({"arm":arm_id,"completed":offset+len(batch),"total":len(cases),"tokens":total_tokens}),flush=True)
    final=tree(model_path)
    if final["sha256"]!=observed["sha256"]: raise ValueError("Model changed during inference")
    verify_protocol()
    write(out/"complete.json",{**binding,"completed_at":utc(),"rows":len(cases),"output_tokens":total_tokens,
          "load_seconds":load_seconds,"generation_seconds":time.monotonic()-generation_started,
          "responses_sha256":sha(rows_path),"model_unchanged":True,"preflight":preflight})
    write(out/"state.json",{"status":"completed","heartbeat_at":utc(),"completed":len(cases)},exclusive=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--arm",required=True)
    p.add_argument("--deadline",type=float,required=True)
    p.add_argument("--preflight",action="store_true")
    args=p.parse_args()
    run(args.work,args.arm,args.deadline,args.preflight)
