from types import SimpleNamespace
import pytest

from ruflo_model.backend import BoundModelProvider


def test_concrete_model_requests_cannot_drop_or_replace_configured_adapter():
    provider = object.__new__(BoundModelProvider)
    provider.cli_args = SimpleNamespace(model="/base", adapter_path="/adapter", draft_model=None)
    provider.model_key = None
    provider.model, provider.tokenizer = object(), object()
    loaded = []
    def load(*key):
        loaded.append(key)
        provider.model_key = key
    provider._load = load
    provider.load("default_model", None, "default_model")
    provider.load("/base")
    assert loaded == [("/base", "/adapter", None)]
    with pytest.raises(ValueError, match="configured adapter"):
        provider.load("/base", "/wrong")
    with pytest.raises(ValueError, match="configured model"):
        provider.load("/wrong")
