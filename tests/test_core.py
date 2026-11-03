import json
from pathlib import Path

from ruflo_model.common import ROOT, digest, read_jsonl
from ruflo_model.data import call, tool_case
from ruflo_model.scoring import score_code, score_tool
from ruflo_model.selection import promote


def test_dataset_is_frozen_and_prompts_disjoint():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    prompts = []
    for file, sha in manifest["files"].items():
        assert digest(ROOT / "data" / file) == sha
        rows = read_jsonl(ROOT / "data" / file)
        this = {json.dumps(r["messages"][:-1] if file.startswith("training") else r["messages"], sort_keys=True) for r in rows}
        assert all(not this.intersection(other) for other in prompts)
        prompts.append(this)


def test_tools_reject_scope_expansion_and_malformed_args():
    case = tool_case("dev", 1, "edit")
    e = case["expected"]
    assert score_tool(case, [e])[0]
    wrong = {"name": e["name"], "arguments": dict(e["arguments"], replace_all=True)}
    assert not score_tool(case, [wrong])[0]
    assert not score_tool(case, [{"name": "Bash", "arguments": {"command": "pwd"}}])[0]
    assert not score_tool(case, [{"name": "Edit", "arguments": "{"}])[0]


def test_code_executes_assertions():
    case = {"tests": "assert f(2)==3; assert f(-1)==0"}
    assert score_code(case, "def f(x):\n return x+1")[0]
    assert not score_code(case, "def f(x):\n return 3")[0]
    assert not score_code(case, "import os\ndef f(x): return x+1")[0]


def test_os_sandbox_blocks_dynamic_file_and_network_access(tmp_path):
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("not a secret")
    code = f'''def blocked():
    try:
        eval("open({str(sentinel)!r}).read()")
    except PermissionError:
        return True
    return False
'''
    assert score_code({"tests": "assert blocked()"}, code)[0]
    network = '''def blocked():
    try:
        eval("__import__('socket').socket().connect(('127.0.0.1', 9110))")
    except PermissionError:
        return True
    return False
'''
    assert score_code({"tests": "assert blocked()"}, network)[0]


def test_promotion_rejects_regressions():
    base = {"total": 10, "passed": 8, "truncated": 0, "output_tokens": 100,
            "groups": {"tool": {"passed": 5, "total": 6}, "code": {"passed": 3, "total": 4}}}
    worse = dict(base, passed=8, output_tokens=80,
                 groups={"tool": {"passed": 6, "total": 6}, "code": {"passed": 2, "total": 4}})
    rule = {"minimum_additional_passes": 1, "allow_latency_tiebreak": True, "max_token_ratio_on_tie": .9}
    assert not promote(base, worse, rule)[0]
    assert promote(base, dict(base, output_tokens=80), rule)[0]
