"""Independent checks for the new synthetic confirmation bundle."""
import ast
import collections
import hashlib
import importlib.util
import json
import operator
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("confirmation_prepare", SOURCE / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def integer_expression(expression):
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        raise AssertionError(f"Unexpected arithmetic syntax: {ast.dump(node)}")

    return visit(ast.parse(expression, mode="eval").body)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.cases = prepare.build_cases()

    def test_determinism_balance_and_no_code_reuse(self):
        self.assertEqual(self.cases, prepare.build_cases())
        self.assertEqual(len(self.cases), 256)
        self.assertEqual(len({case["id"] for case in self.cases}), 256)
        self.assertEqual(collections.Counter(case["kind"] for case in self.cases), {"tool": 192, "math": 64})
        self.assertEqual(collections.Counter(case["family"] for case in self.cases if case["kind"] == "tool"), dict.fromkeys(("read", "slice", "glob", "grep", "write", "edit", "injection", "test"), 24))
        for case in self.cases:
            self.assertNotIn("HumanEval", case["id"])
            label_key = "expected" if case["kind"] == "tool" else "answer"
            self.assertEqual(set(case), {"id", "kind", "family", "messages", "tools", label_key})
            self.assertNotIn("expected", json.dumps(case["messages"]))

    def test_arithmetic_labels_recomputed_from_visible_expressions(self):
        answers = []
        for case in self.cases:
            if case["kind"] == "math":
                expression = case["messages"][-1]["content"].split(": evaluate ", 1)[1].removesuffix(".")
                self.assertEqual(integer_expression(expression), case["answer"])
                self.assertIs(type(case["answer"]), int)
                self.assertEqual(case["tools"], [])
                answers.append(case["answer"])
        self.assertGreater(len(set(answers)), 50)

    def test_tool_labels_are_scoped_and_supported_by_visible_request(self):
        for case in self.cases:
            if case["kind"] != "tool":
                continue
            expected = case["expected"]
            name, args = expected["name"], expected["arguments"]
            visible = "\n".join(message["content"] for message in case["messages"])
            definitions = {item["function"]["name"]: item["function"]["parameters"] for item in case["tools"]}
            self.assertEqual(set(definitions), {"Read", "Write", "Edit", "Glob", "Grep", "Bash"})
            schema = definitions[name]
            self.assertLessEqual(set(schema["required"]), set(args))
            self.assertLessEqual(set(args), set(schema["properties"]))
            for key, value in args.items():
                if key == "output_mode":
                    self.assertEqual(value, "content")
                    self.assertIn("Display matching source lines", visible)
                else:
                    self.assertIn(str(value), visible)
                if key in ("path", "file_path"):
                    self.assertTrue(value.startswith("/workspace/confirmation_v1/"))
            if case["family"] in ("edit", "injection"):
                observed = next(message["content"] for message in case["messages"] if message["role"] == "tool")
                self.assertEqual(observed.count(args["old_string"]), 1)
                self.assertNotIn(args["new_string"], observed)
                self.assertEqual(name, "Edit")
                self.assertNotIn("replace_all", args)
                edited = observed.replace(args["old_string"], args["new_string"], 1)
                self.assertEqual(len(edited.splitlines()), len(observed.splitlines()))
                self.assertIn("_enabled = True", edited)
                if case["family"] == "injection":
                    self.assertIn("untrusted.invalid", observed)
                    self.assertIn("untrusted.invalid", edited)
                    self.assertNotIn("untrusted.invalid", json.dumps(args))
            if name == "Write":
                self.assertTrue(args["content"].endswith("\n"))
                ast.parse(args["content"])
            if name == "Bash":
                self.assertRegex(args["command"], r"^python -m pytest /workspace/confirmation_v1/test/[a-z0-9_]+/tests/test_[a-z0-9_]+\.py -q$")

    def test_messages_are_exactly_disjoint_from_repository_corpora(self):
        evidence = prepare.assert_disjoint(self.cases)
        self.assertGreater(evidence["repository_sources"], 3)
        self.assertGreater(evidence["repository_message_sequences"], 100)
        self.assertEqual(evidence["exact_overlaps"], 0)

    def test_disjointness_rejects_reuse_in_jsonl_and_nested_research(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "research/run").mkdir(parents=True)
            training = root / "data/train.jsonl"
            training.write_text(json.dumps({"messages": self.cases[0]["messages"]}) + "\n")
            with self.assertRaisesRegex(ValueError, "reuse"):
                prepare.assert_disjoint(self.cases, root)
            training.unlink()
            (root / "research/run/prompts.json").write_text(json.dumps({"prompts": [self.cases[2]]}))
            with self.assertRaisesRegex(ValueError, "reuse"):
                prepare.assert_disjoint(self.cases, root)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                prepare.assert_disjoint([self.cases[1], self.cases[1]], root)

    def make_fixture(self, root):
        models = root / "research/vast_cross_model"
        models.mkdir(parents=True)
        (models / "models.json").write_text(json.dumps([{"key": "qwen035-4b-base", "model": "Qwen/Qwen3.5-4B", "revision": prepare.MODEL_REVISION, "weights_sha256": {"model.safetensors": "b" * 64}}]))
        adapter = root / "weights/coding-4b"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"fixture adapter")
        (adapter / "adapter_config.json").write_text("{}\n")
        source = root / "sources"
        source.mkdir()
        for name in ("worker.py", "measurement.py", "prepare.py"):
            (source / name).write_text(f"# {name}\n")
        (source / "model-assets.json").write_text(json.dumps({"repo_id": "Qwen/Qwen3.5-4B", "revision": prepare.MODEL_REVISION, "asset_files": {"config.json": "c" * 64}}))
        return source

    def test_bundle_hashes_reproducibility_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_fixture(root)
            out = root / "bundle"
            prepare.prepare(out, root=root, source=source)
            self.assertEqual({item.name for item in out.iterdir()}, {"cases.json", "protocol.json"})
            protocol = json.loads((out / "protocol.json").read_text())
            self.assertEqual(protocol["schema"], "ruvllm-model-measurement/v1")
            self.assertEqual(protocol["run_id"], prepare.RUN_ID)
            self.assertEqual(protocol["case_count"], 256)
            self.assertEqual(protocol["cases_sha256"], hashlib.sha256((out / "cases.json").read_bytes()).hexdigest())
            self.assertEqual(protocol["model"]["asset_files"], {"config.json": "c" * 64})
            self.assertEqual(protocol["adapter"]["weights_sha256"], hashlib.sha256(b"fixture adapter").hexdigest())
            for name, digest in protocol["source_sha256"].items():
                self.assertEqual(digest, hashlib.sha256((source / name).read_bytes()).hexdigest())
            before = {item.name: item.read_bytes() for item in out.iterdir()}
            with self.assertRaises(FileExistsError):
                prepare.prepare(out, root=root, source=source)
            self.assertEqual(before, {item.name: item.read_bytes() for item in out.iterdir()})
            again = root / "again"
            prepare.prepare(again, root=root, source=source)
            self.assertEqual(before, {item.name: item.read_bytes() for item in again.iterdir()})

    def test_rejects_relative_and_symlink_outputs_and_missing_sources(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            prepare.prepare(Path("relative"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_fixture(root)
            (root / "alias").symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic"):
                prepare.prepare(root / "alias/bundle", root=root, source=source)
            (source / "worker.py").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare.prepare(root / "bundle", root=root, source=source)
            self.assertFalse((root / "bundle").exists())
            with self.assertRaisesRegex(ValueError, "run_id"):
                prepare.build_protocol(b"[]", root=root, source=source, run_id="bad/run")


if __name__ == "__main__":
    unittest.main()
