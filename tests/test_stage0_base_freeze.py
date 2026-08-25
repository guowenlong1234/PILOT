import torch

from vlnce_baselines.nwm.active_lookahead.base_freeze import (
    capture_base_tensor_manifest,
    compare_base_tensor_manifests,
)


def test_base_tensor_manifest_is_exact_and_detects_one_changed_tensor():
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    before = capture_base_tensor_manifest({"base": module})
    same = capture_base_tensor_manifest({"base": module})
    assert compare_base_tensor_manifests(before, same)["exact_match"] is True

    with torch.no_grad():
        module[0].weight[0, 0].add_(1)
    after = capture_base_tensor_manifest({"base": module})
    result = compare_base_tensor_manifests(before, after)
    assert result["exact_match"] is False
    assert result["changed"] == ["base.0.weight"]


def test_base_tensor_manifest_includes_buffers_and_unusual_dtypes():
    module = torch.nn.BatchNorm1d(3).to(dtype=torch.bfloat16)
    manifest = capture_base_tensor_manifest({"base": module})
    assert "base.running_mean" in manifest["tensors"]
    assert manifest["tensors"]["base.weight"]["dtype"] == "torch.bfloat16"

