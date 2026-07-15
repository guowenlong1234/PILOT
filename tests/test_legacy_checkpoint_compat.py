import io

import pytest

from vlnce_baselines.common import checkpoint_compat


def test_legacy_unpickler_only_replaces_old_habitat_config():
    unpickler = checkpoint_compat.LegacyCheckpointUnpickler(io.BytesIO())
    assert (
        unpickler.find_class("habitat.config.default", "Config")
        is checkpoint_compat.LegacyHabitatConfig
    )
    assert unpickler.find_class("builtins", "dict") is dict


@pytest.mark.parametrize("error", [KeyError("_content"), RecursionError()])
def test_loader_retries_legacy_config_failures(monkeypatch, error):
    calls = []

    def fake_load(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        if len(calls) == 1:
            raise error
        return {"state_dict": {"weight": 1}}

    monkeypatch.setattr(checkpoint_compat.torch, "load", fake_load)
    result = checkpoint_compat.load_torch_checkpoint_compat(
        "legacy.pth",
        map_location="cpu",
    )

    assert result == {"state_dict": {"weight": 1}}
    assert "pickle_module" not in calls[0][2]
    assert (
        calls[1][2]["pickle_module"]
        is checkpoint_compat.LegacyCheckpointPickleModule
    )


def test_loader_does_not_hide_unrelated_errors(monkeypatch):
    def fake_load(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(checkpoint_compat.torch, "load", fake_load)
    with pytest.raises(FileNotFoundError, match="missing"):
        checkpoint_compat.load_torch_checkpoint_compat("missing.pth")
