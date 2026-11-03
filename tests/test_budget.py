import pytest
from ruflo_model.cloud import Budget, CAP_MICRO_USD


def test_budget_reserves_before_work_and_never_exceeds_cap(tmp_path):
    budget = Budget(tmp_path / "budget.json")
    first = budget.reserve(3_000_000, {"purpose": "test"})
    budget.annotate(first["id"], status="submission_failed_or_unknown")
    with pytest.raises(ValueError, match="Reconcile"):
        budget.reserve(1, {})
    budget.annotate(first["id"], status="reconciled_no_job")
    with pytest.raises(ValueError, match="immutable"):
        budget.annotate(first["id"], reserved_micro_usd=0)
    with pytest.raises(ValueError, match="exhausted"):
        budget.reserve(CAP_MICRO_USD - 2_999_999, {})
    budget.reserve(CAP_MICRO_USD - 3_000_000, {})
    with pytest.raises(ValueError):
        budget.reserve(1, {})
