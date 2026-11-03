import unittest
from common import ROOT, jsonl
from prepare import math_cases
from score import grade, paired, parse_tools
from ruflo_model.scoring import score_tool


class ProtocolTests(unittest.TestCase):
    def test_provider_credentials_are_removed(self):
        from common import clean_environment
        from unittest.mock import patch
        with patch.dict("os.environ",{"CONTAINER_API_KEY":"test-value","HF_TOKEN":"test-value","PATH":"/usr/bin","CUDA_VISIBLE_DEVICES":"0"},clear=True):
            env=clean_environment()
            self.assertNotIn("CONTAINER_API_KEY",env)
            self.assertNotIn("HF_TOKEN",env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"],"0")
            self.assertEqual(env["HF_TOKEN_PATH"],"/dev/null")

    def test_changed_runtime_receipt_is_rejected(self):
        import tempfile
        from pathlib import Path
        from common import HERE, object_sha, read, sha, write
        from score import verify_binding
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)
            folder=work/"outputs/arm"
            spec=read(HERE/"models.json")[0]
            arm={"id":"test-arm","variant":spec["key"]}
            files=dict(spec["weights_sha256"])
            manifest={"variant":spec["key"],"source":spec,"files":files,"sha256":object_sha(files)}
            path=work/"models"/(spec["key"]+".manifest.json")
            write(path,manifest)
            runtime={"versions":{"vllm":"0.21.0"}}
            write(folder/"runtime.json",runtime)
            binding={"protocol_sha256":"a"*64,"model_sha256":manifest["sha256"],"model_manifest_sha256":sha(path),
                     "runtime_sha256":object_sha(runtime),"variant":spec["key"],"arm":arm["id"]}
            write(folder/"binding.json",binding)
            protocol={"runtime_versions":{"vllm":"0.21.0"}}
            self.assertEqual(verify_binding(work,folder,arm,protocol,"a"*64,binding),binding)
            with self.assertRaises(ValueError):verify_binding(work,folder,arm,protocol,"a"*64,{**binding,"model_sha256":"b"*64})
            write(folder/"runtime.json",{"versions":{"vllm":"changed"}},exclusive=False)
            with self.assertRaises(ValueError):verify_binding(work,folder,arm,protocol,"a"*64,binding)
    def test_complete_answer_is_required(self):
        case={"id":"case","panel":"math128","kind":"math","answer":42}
        row={"text":"Work \\boxed{42}","thinking":True,"finish_reason":"length","output_tokens":8192}
        result=grade(case,row)
        self.assertTrue(result["answer_correct"])
        self.assertFalse(result["completed"])
        self.assertFalse(result["delivered_correct"])
        row.update(text="Work </think> The result is \\boxed{42}<|im_end|>",finish_reason="stop")
        self.assertTrue(grade(case,row)["delivered_correct"])

    def test_box_in_reasoning_does_not_count_as_delivered_answer(self):
        case={"id":"case","panel":"math128","kind":"math","answer":42}
        row={"text":"\\boxed{42}</think>No answer.","thinking":True,"finish_reason":"stop","output_tokens":40}
        result=grade(case,row)
        self.assertTrue(result["answer_correct"])
        self.assertTrue(result["completed"])
        self.assertFalse(result["delivered_correct"])

    def test_original_tool_reference_roundtrip(self):
        import json
        cases=[c for c in jsonl(ROOT/"data/release.jsonl") if c["kind"]=="tool"]
        self.assertEqual(len(cases),27)
        for case in cases:
            expected=case["expected"]
            args="".join(f"<parameter={k}>\n{v if isinstance(v,str) else json.dumps(v)}\n</parameter>" for k,v in expected["arguments"].items())
            text=f"<tool_call><function={expected['name']}>{args}</function></tool_call>"
            self.assertTrue(score_tool(case,parse_tools(text,case["tools"]))[0],case["id"])

    def test_tool_duplicate_rejected(self):
        case=jsonl(ROOT/"data/release.jsonl")[0]
        text="<tool_call><function=Read><parameter=file_path>/a</parameter><parameter=file_path>/b</parameter></function></tool_call>"
        with self.assertRaises(ValueError): parse_tools(text,case["tools"])
        text='<tool_call>{"name":"Read","arguments":{"file_path":"/a","file_path":"/b"}}</tool_call>'
        with self.assertRaises(ValueError): parse_tools(text,case["tools"])

    def test_incomplete_call_rejected(self):
        case=jsonl(ROOT/"data/release.jsonl")[0]
        with self.assertRaises(ValueError): parse_tools("<tool_call>{}",case["tools"])

    def test_math_fixtures(self):
        from collections import Counter
        cases=math_cases()
        self.assertEqual(len(cases),128)
        self.assertEqual(set(Counter(c["family"] for c in cases).values()),{8})
        self.assertTrue(all(isinstance(c["answer"],int) for c in cases))
        # Different formulations supply independent checks for selected families.
        for case in cases:
            a,b,c,n=case["parameters"]
            family=int(case["family"].split("-")[-1])
            if family==4:
                import itertools
                self.assertEqual(case["answer"],len(list(itertools.combinations(range(n),3))))
            if family==10:self.assertEqual(case["answer"],sum(range(a,a+n+1)))
            if family==15:
                k=case["answer"]
                self.assertLessEqual(k*b,a*b+c)
                self.assertGreater((k+1)*b,a*b+c)

    def test_paired_direction(self):
        before=[{"id":str(i),"delivered_correct":v} for i,v in enumerate([False,False,True,True])]
        after=[{"id":str(i),"delivered_correct":v} for i,v in enumerate([True,True,False,True])]
        result=paired(before,after)
        self.assertEqual((result["gains"],result["regressions"],result["net"]),(2,1,1))
        self.assertEqual(result["exact_mcnemar_two_sided_p"],1)


if __name__=="__main__":unittest.main()
