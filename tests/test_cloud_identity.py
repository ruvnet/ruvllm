"""Cloud identity tests use fake provider clients and temporary local state."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from ruflo_model import cloud, cloud_train


def receipt(root, name="cloud", **fields):
    path = root / "runtime" / name / "job.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"job_id": "fixture-job", "reserved_max_usd": 1, **fields}))
    return path


def api_fixture():
    api = Mock()
    api.whoami.return_value = {"name": "authenticated-owner"}
    api.run_job.return_value = SimpleNamespace(id="fixture-job", url="https://huggingface.co/jobs/fixture")
    api.inspect_job.return_value = SimpleNamespace(id="fixture-job", status=SimpleNamespace(stage="COMPLETED"))
    api.fetch_job_logs.return_value = []
    return api


def test_benchmark_submission_uses_authenticated_namespace_before_reservation(tmp_path, monkeypatch):
    api = api_fixture()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    budget = cloud.Budget(tmp_path / "budget.json")
    monkeypatch.setattr(cloud, "Budget", lambda: budget)
    (tmp_path / "data").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "data/dev.jsonl").write_text(json.dumps({"id": "fixture", "messages": [], "tools": []}) + "\n")
    (tmp_path / "scripts/cloud_benchmark.py").write_text("print('fixture')\n")
    hardware = Mock()
    hardware.json.return_value = [{"name": "a100-large", "unitLabel": "minute", "unitCostMicroUSD": 1000}]
    monkeypatch.setattr(cloud.httpx, "get", lambda *args, **kwargs: hardware)
    observed = cloud.submit_benchmark()
    assert observed["namespace"] == "authenticated-owner"
    assert api.run_job.call_args.kwargs["namespace"] == "authenticated-owner"
    assert api.run_job.call_args.kwargs["secrets"] == {}
    assert json.loads(budget.path.read_text())["reservations"][0]["namespace"] == "authenticated-owner"


def test_invalid_authenticated_identity_precedes_price_or_reservation(tmp_path, monkeypatch):
    api = api_fixture()
    api.whoami.return_value = {"name": "../untrusted"}
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    hardware = Mock(side_effect=AssertionError("No provider hardware call expected"))
    monkeypatch.setattr(cloud.httpx, "get", hardware)
    with pytest.raises(ValueError, match="namespace"):
        cloud.submit_benchmark()
    hardware.assert_not_called()
    api.run_job.assert_not_called()
    assert not (tmp_path / "runtime/cloud-budget.json").exists()


@pytest.mark.parametrize("operation", [cloud.inspect, cloud.cancel])
def test_existing_benchmark_requires_receipt_namespace_before_client_creation(tmp_path, monkeypatch, operation):
    receipt(tmp_path)
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    client = Mock(side_effect=AssertionError("Missing namespace must fail before provider access"))
    monkeypatch.setattr(huggingface_hub, "HfApi", client)
    with pytest.raises(ValueError, match="missing namespace"):
        operation()
    client.assert_not_called()


@pytest.mark.parametrize("operation", [cloud.inspect, cloud.cancel])
def test_existing_benchmark_keeps_recorded_namespace(tmp_path, monkeypatch, operation):
    receipt(tmp_path, namespace="recorded-owner")
    api = api_fixture()
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    observed = operation()
    assert observed["namespace"] == "recorded-owner"
    api.inspect_job.assert_called_once_with(job_id="fixture-job", namespace="recorded-owner")
    api.fetch_job_logs.assert_called_once_with(job_id="fixture-job", namespace="recorded-owner", follow=False)
    api.whoami.assert_not_called()
    if operation is cloud.cancel:
        api.cancel_job.assert_called_once_with(job_id="fixture-job", namespace="recorded-owner")


@pytest.mark.parametrize("namespace", [None, "", "../account", "account/other", {"name": "account"}, True])
def test_receipt_rejects_invalid_namespace(namespace):
    with pytest.raises(ValueError, match="namespace"):
        cloud.receipt_namespace({"namespace": namespace})


@pytest.mark.parametrize("fields", [{"bucket": "recorded-owner/fixture"},
                                    {"namespace": "recorded-owner", "bucket": "other-owner/fixture"}])
def test_training_rejects_missing_or_mismatched_receipt_identity(tmp_path, monkeypatch, fields):
    receipt(tmp_path, "cloud-train", **fields)
    api = api_fixture()
    monkeypatch.setattr(cloud_train, "ROOT", tmp_path)
    monkeypatch.setattr(cloud_train, "HfApi", lambda: api)
    monkeypatch.setattr(cloud_train, "Budget", Mock())
    with pytest.raises(ValueError, match="namespace"):
        cloud_train.run(tmp_path / "unused")
    assert not api.mock_calls


def test_training_resume_uses_receipt_for_inspection_logs_and_bucket(tmp_path, monkeypatch):
    receipt(tmp_path, "cloud-train", namespace="recorded-owner", bucket="recorded-owner/fixture")
    api = api_fixture()
    monkeypatch.setattr(cloud_train, "ROOT", tmp_path)
    monkeypatch.setattr(cloud_train, "HfApi", lambda: api)
    monkeypatch.setattr(cloud_train, "Budget", Mock())
    monkeypatch.setattr(cloud_train.signal, "signal", lambda *args: None)
    monkeypatch.setattr(cloud_train, "convert", lambda *args: [])
    cloud_train.run(tmp_path / "converted")
    api.inspect_job.assert_called_once_with(job_id="fixture-job", namespace="recorded-owner")
    api.fetch_job_logs.assert_called_once_with(job_id="fixture-job", namespace="recorded-owner", follow=False)
    api.sync_bucket.assert_called_once_with(source="hf://buckets/recorded-owner/fixture/training-v1",
                                           dest=str(tmp_path / "runtime/cloud-train/download"), quiet=True)
    api.whoami.assert_not_called()
    api.run_job.assert_not_called()
