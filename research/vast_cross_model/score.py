"""Independent local scoring of immutable raw generations. No cloud code execution."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from common import HERE, ROOT, jsonl, object_sha, read, sha, verify_protocol, write
sys.path.insert(0,str(ROOT))
from ruflo_model.scoring import score_code, score_tool


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError("Duplicate JSON argument")
        result[key]=value
    return result


def parse_tools(text,tools):
    blocks=re.findall(r"<tool_call>(.*?)</tool_call>",text,re.S)
    if text.count("<tool_call>")!=len(blocks) or text.count("</tool_call>")!=len(blocks):
        raise ValueError("Incomplete tool envelope")
    calls=[]
    for block in blocks:
        block=block.strip()
        if block.startswith("{"):
            value=json.loads(block,object_pairs_hook=unique_object)
            if set(value)!={"name","arguments"}: raise ValueError("Unexpected JSON tool fields")
            calls.append(value)
            continue
        match=re.fullmatch(r"<function=([^<>]+)>(.*?)</function>",block,re.S)
        if not match: raise ValueError("Malformed function envelope")
        name,body=match.groups()
        schema=next(t["function"]["parameters"]["properties"] for t in tools if t["function"]["name"]==name)
        args={}
        previous=0
        for param in re.finditer(r"<parameter=([^<>]+)>(.*?)</parameter>",body,re.S):
            if body[previous:param.start()].strip(): raise ValueError("Unparsed argument content")
            key,value=param.groups()
            if key in args or key not in schema: raise ValueError("Duplicate or unknown parameter")
            value=value.removeprefix("\n").removesuffix("\n")
            kind=schema[key].get("type","string")
            if kind=="integer":
                if not re.fullmatch(r"-?\d+",value): raise ValueError("Invalid integer")
                value=int(value)
            elif kind=="boolean":
                if value.lower() not in ("true","false"): raise ValueError("Invalid Boolean")
                value=value.lower()=="true"
            elif kind!="string":
                value=json.loads(value,object_pairs_hook=unique_object)
            args[key]=value
            previous=param.end()
        if body[previous:].strip(): raise ValueError("Trailing argument content")
        calls.append({"name":name,"arguments":args})
    return calls


def grade(case,row):
    text=row["text"]
    closed=not row["thinking"] or "</think>" in text
    if text.rfind("<think>")>text.rfind("</think>"): closed=False
    completed=row["finish_reason"]=="stop" and closed
    visible=text.rsplit("</think>",1)[-1] if "</think>" in text else text
    visible=re.sub(r"<\|(?:im_end|endoftext)\|>","",visible).strip()
    try:
        if case["kind"]=="math":
            values=re.findall(r"\\boxed\{\s*([+-]?\d+)\s*\}",text)
            correct=bool(values) and int(values[-1])==case["answer"]
            # A correct answer inside unfinished reasoning is observed but undelivered.
            final_values=re.findall(r"\\boxed\{\s*([+-]?\d+)\s*\}",visible)
            delivered=completed and bool(final_values) and int(final_values[-1])==case["answer"]
            reason="integer exact match" if correct else "missing or incorrect final boxed integer"
        elif case["kind"]=="code":
            correct,reason=score_code(case,visible)
            delivered=completed and correct
        else:
            correct,reason=score_tool(case,parse_tools(visible,case["tools"]))
            delivered=completed and correct
    except (ValueError,TypeError,KeyError,StopIteration) as error:
        correct,delivered,reason=False,False,f"strict parse: {type(error).__name__}"
    return {"id":case["id"],"panel":case["panel"],"answer_correct":bool(correct),
            "completed":bool(completed),"delivered_correct":bool(delivered),"reason":reason,
            "truncated":row["finish_reason"]=="length","output_tokens":row["output_tokens"]}


def summarize(rows):
    return {"count":len(rows),**{k:sum(r[k] for r in rows) for k in
            ("answer_correct","completed","delivered_correct","truncated","output_tokens")},
            "delivered_accuracy":sum(r["delivered_correct"] for r in rows)/len(rows)}


def paired(left,right):
    assert [r["id"] for r in left]==[r["id"] for r in right]
    gains=sum(not a["delivered_correct"] and b["delivered_correct"] for a,b in zip(left,right))
    losses=sum(a["delivered_correct"] and not b["delivered_correct"] for a,b in zip(left,right))
    discordant=gains+losses
    p=min(1.0,2*sum(math.comb(discordant,i) for i in range(min(gains,losses)+1))/2**discordant) if discordant else 1.0
    return {"gains":gains,"regressions":losses,"net":gains-losses,"exact_mcnemar_two_sided_p":p,
            "multiple_testing_adjusted":False,"exploratory":True}


def verify_binding(work,folder,arm,protocol,protocol_sha,complete):
    binding=read(folder/"binding.json")
    if set(binding)!={"protocol_sha256","model_sha256","model_manifest_sha256","runtime_sha256","variant","arm"}:
        raise ValueError("Unexpected binding keys")
    if binding["arm"]!=arm["id"] or binding["variant"]!=arm["variant"] or binding["protocol_sha256"]!=protocol_sha:
        raise ValueError("Binding does not match frozen arm")
    if any(complete.get(k)!=v for k,v in binding.items()): raise ValueError("Completion binding differs")
    actual_runtime=read(folder/"runtime.json")
    if object_sha(actual_runtime)!=binding["runtime_sha256"]: raise ValueError("Runtime receipt changed")
    if any(actual_runtime["versions"][k].split("+")[0]!=v for k,v in protocol["runtime_versions"].items()):
        raise ValueError("Runtime versions differ")
    manifest_path=work/"models"/(arm["variant"]+".manifest.json")
    if sha(manifest_path)!=binding["model_manifest_sha256"]: raise ValueError("Model manifest changed")
    manifest=read(manifest_path)
    if manifest["variant"]!=arm["variant"] or object_sha(manifest["files"])!=binding["model_sha256"] or manifest["sha256"]!=binding["model_sha256"]:
        raise ValueError("Model digest or variant mismatch")
    pins={p["key"]:p for p in read(HERE/"models.json")}
    if arm["variant"] in pins:
        spec=pins[arm["variant"]]
        if manifest["source"]!=spec or any(manifest["files"].get(k)!=v for k,v in spec["weights_sha256"].items()):
            raise ValueError("Original model no longer matches pinned source")
    elif arm["variant"]=="qwen035-08b-edited":
        base=read(work/"models/qwen035-08b-base.manifest.json")
        weight="model.safetensors-00001-of-00001.safetensors"
        changed={k for k in set(base["files"])|set(manifest["files"]) if base["files"].get(k)!=manifest["files"].get(k)}
        if manifest["base_sha256"]!=base["sha256"] or changed!={weight}: raise ValueError("Native candidate changed unrelated files")
        if manifest["files"][weight]!="bd5c223dc7d0ed48a6857f798a94da6ec46c40b9ff4e7bc31872dba10f943ae5":
            raise ValueError("Native published edit differs")
        assets=read(HERE/"assets.json")
        if manifest["patch_sha256"]!=assets["patch.json"] or manifest["policy_sha256"]!=assets["policy.json"]:
            raise ValueError("Native edit input binding differs")
    elif arm["variant"]=="qwen035-4b-lora":
        base=read(work/"models/qwen035-4b-base.manifest.json")
        if manifest["base_sha256"]!=base["sha256"] or manifest["adapter_sha256"]!=read(HERE/"assets.json")["adapter_model.safetensors"]:
            raise ValueError("LoRA lineage differs")
        expected={f"model.language_model.layers.{n}.mlp.down_proj.weight" for n in range(16,32)}
        if {r["tensor"] for r in manifest["changed_tensors"]}!=expected or len(manifest["changed_tensors"])!=16:
            raise ValueError("LoRA target set differs")
        if manifest["untouched_tensor_hashes_verified"]<=0: raise ValueError("Missing untouched tensor audit")
        if set(base["files"])!=set(manifest["files"]): raise ValueError("Unexpected LoRA model files")
        if any(v!=manifest["files"][k] for k,v in base["files"].items() if not k.endswith(".safetensors")):
            raise ValueError("LoRA changed tokenizer or configuration")
    else: raise ValueError("Unrecognized variant")
    return binding


def score(work,out):
    protocol,protocol_sha=verify_protocol()
    cases={c["id"]:c for c in read(HERE/"cases.json")}
    out.mkdir(parents=True,exist_ok=False)
    scored={}
    summaries={}
    for arm in protocol["arms"]:
        folder=work/"outputs"/arm["id"]
        complete=read(folder/"complete.json")
        binding=verify_binding(work,folder,arm,protocol,protocol_sha,complete)
        assert complete["protocol_sha256"]==protocol_sha and complete["model_unchanged"] and not complete["preflight"]
        assert sha(folder/"responses.jsonl")==complete["responses_sha256"]
        rows=jsonl(folder/"responses.jsonl")
        expected=[c["id"] for c in cases.values() if c["panel"] in arm["panels"]]
        assert [r["id"] for r in rows]==expected and len(rows)==complete["rows"]
        results=[]
        for row in rows:
            assert all(row[k]==v for k,v in binding.items())
            assert row["thinking"]==arm["thinking"]
            assert row["kind"]==cases[row["id"]]["kind"] and row["panel"]==cases[row["id"]]["panel"]
            assert row["finish_reason"] in ("stop","length")
            assert row["max_tokens"]==protocol["panel_max_tokens"][row["panel"]]
            assert len(row["token_ids"])==row["output_tokens"] and 0<row["output_tokens"]<=row["max_tokens"]
            assert row["finish_reason"]!="length" or row["output_tokens"]==row["max_tokens"]
            results.append(grade(cases[row["id"]],row))
        assert sum(r["output_tokens"] for r in rows)==complete["output_tokens"]
        scored[arm["id"]]=results
        summaries[arm["id"]]={"panels":{panel:summarize([r for r in results if r["panel"]==panel]) for panel in arm["panels"]},
                              "generation_seconds":complete["generation_seconds"],"load_seconds":complete["load_seconds"],
                              "responses_sha256":complete["responses_sha256"],"binding":binding}
    comparisons={}
    for baseline,candidate in protocol["comparisons"]:
        panels=next(a["panels"] for a in protocol["arms"] if a["id"]==baseline)
        comparisons[candidate]={p:paired([r for r in scored[baseline] if r["panel"]==p],
                                        [r for r in scored[candidate] if r["panel"]==p]) for p in panels}
    count=sum(len(r) for r in scored.values())
    assert count==protocol["total_generations"]
    result={"schema":"ruflo.vast-cross-model-results.v1","protocol_sha256":protocol_sha,"generations":count,
            "selection_use":False,"sota_verified":False,"production_qualified":False,"arms":summaries,"paired":comparisons,
            "limitations":protocol["limitations"],"scorer_sha256":sha(HERE/"score.py"),
            "sandbox_scorer_sha256":sha(ROOT/"ruflo_model/scoring.py")}
    write(out/"scored.json",scored)
    write(out/"summary.json",result)
    lines=["# Frozen cloud model comparison","",f"Protocol `{protocol_sha}`. {count} generations. No candidate selection or promotion.","",
           "| Arm | Panel | Delivered correct | Completed | Truncated | Output tokens |","| :--- | :--- | ---: | ---: | ---: | ---: |"]
    for name,summary in summaries.items():
        for panel,row in summary["panels"].items():
            lines.append(f"| {name} | {panel} | {row['delivered_correct']}/{row['count']} | {row['completed']} | {row['truncated']} | {row['output_tokens']} |")
    lines += ["","## Limits",""]+protocol["limitations"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"generations":count,"summary":str(out/"summary.json")}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    score(args.work,args.output)
