import ast
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile

import jsonschema


def score_tool(case, calls):
    if len(calls) != 1:
        return False, "expected exactly one tool call"
    actual = calls[0].get("function", calls[0])
    expected = case["expected"]
    if actual.get("name") != expected["name"]:
        return False, "wrong tool"
    args = actual.get("arguments", {})
    try:
        args = json.loads(args) if isinstance(args, str) else args
        schema = next(t["function"]["parameters"] for t in case["tools"] if t["function"]["name"] == expected["name"])
        jsonschema.validate(args, schema)
        if any(args.get(k) != v for k, v in expected["arguments"].items()):
            return False, "wrong required argument"
        # Optional defaults cannot silently expand the authorized action.
        allowed_extra = {"description", "timeout", "-n"}
        extras = set(args) - set(expected["arguments"])
        if "replace_all" in extras and args["replace_all"] is False:
            extras.remove("replace_all")
        if extras - allowed_extra:
            return False, "unrequested argument changes action scope"
    except (ValueError, TypeError, jsonschema.ValidationError):
        return False, "invalid tool arguments"
    return True, "passed"


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks[0] if blocks else text.strip()


def score_code(case, text):
    """Run in a macOS Seatbelt sandbox with no network, secrets, or filesystem writes.

    Tests are separate from model context. A fail closed AST filter restricts
    imports; Seatbelt supplies the OS boundary. Never run this unguarded on Linux.
    """
    code = extract_code(text)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "syntax error"
    safe_modules = {"collections", "copy", "math", "functools", "itertools", "heapq", "bisect", "re", "typing", "hashlib"}
    forbidden_names = {"exec", "compile", "open", "input", "__import__", "globals", "locals", "getattr", "setattr", "delattr", "vars", "breakpoint"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(n.name.split('.')[0] not in safe_modules for n in node.names):
            return False, "unsupported import"
        if isinstance(node, ast.ImportFrom) and (node.module or "").split('.')[0] not in safe_modules:
            return False, "unsupported import"
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            return False, "unsupported dynamic operation"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "unsupported introspection"
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists():
        return False, "sandbox unavailable"
    with tempfile.TemporaryDirectory(prefix="ruflo-eval-") as temp:
        path = Path(temp).resolve() / "candidate.py"
        path.write_text(code + "\n\n" + case["tests"] + "\n")
        base = Path(sys.base_prefix).resolve()
        # Read access only to runtime libraries and this fixture. Everything else,
        # including network and process creation by Python, remains denied.
        profile = f'''(version 1)
(deny default)
(allow process-exec (literal "{Path(sys.executable).resolve()}"))
(allow sysctl-read)
(allow file-read-metadata)
(allow file-read*
 (require-all (require-not (subpath "/Users"))
              (require-not (subpath "/private"))
              (require-not (subpath "/Volumes")))
 (subpath "{base}") (subpath "{Path(sys.prefix).resolve()}")
 (subpath "{Path(temp).resolve()}"))
'''
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        try:
            result = subprocess.run(["/usr/bin/sandbox-exec", "-p", profile, sys.executable,
                                     "-I", "-S", "-B", str(path)],
                                    cwd=temp, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    timeout=4, preexec_fn=limits)
            return result.returncode == 0, "passed" if result.returncode == 0 else "execution or assertion failure"
        except subprocess.TimeoutExpired:
            return False, "execution timeout"


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["kind"], []).append(row)
    return {"total": len(rows), "passed": sum(r["passed"] for r in rows),
            "accuracy": sum(r["passed"] for r in rows) / max(1, len(rows)),
            "truncated": sum(r["finish_reason"] == "length" for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "elapsed_seconds": sum(r["seconds"] for r in rows),
            "groups": {k: {"passed": sum(r["passed"] for r in v), "total": len(v)} for k, v in groups.items()}}
