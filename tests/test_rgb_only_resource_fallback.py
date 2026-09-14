import json
import subprocess

import pytest

from scripts import rgb_only_optimization as workflow


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_server_nvml_version_mismatch_uses_runtime_cuda_probe(monkeypatch, capsys):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "nvidia-smi":
            return completed(1, stderr="Failed to initialize NVML: Driver/library version mismatch\n"
                                         "NVML library version: 580.178")
        payload = {"torch": "2.2.2+cu121", "cuda": "12.1", "devices": [
            {"visible_index": 0, "name": "NVIDIA RTX A6000", "memory_used_mib": 320,
             "memory_total_mib": 48640},
            {"visible_index": 1, "name": "NVIDIA RTX A6000", "memory_used_mib": 300,
             "memory_total_mib": 48640},
        ]}
        return completed(stdout=json.dumps(payload) + "\n")

    monkeypatch.setattr(workflow.subprocess, "run", run)
    with workflow.resources("server", "0,1"):
        pass

    assert calls[1] == workflow.runtime("server", ["-c", workflow.CUDA_RESOURCE_FALLBACK], "0,1")
    assert "nvidia_smi_fallback=runtime_cuda" in capsys.readouterr().out


def test_non_version_mismatch_does_not_fallback(monkeypatch):
    monkeypatch.setattr(workflow.subprocess, "run",
                        lambda *args, **kwargs: completed(1, stderr="No devices were found"))
    with pytest.raises(RuntimeError, match="nvidia-smi resource check failed"):
        with workflow.resources("server", "0"):
            pass


def test_eval_machine_does_not_use_server_fallback(monkeypatch):
    monkeypatch.setattr(workflow, "output", lambda command: "true")
    def run(command, **kwargs):
        if command[:3] == ["docker", "exec", "gwl-etpnav"]:
            return completed()
        return completed(1, stderr="Failed to initialize NVML: Driver/library version mismatch")

    monkeypatch.setattr(workflow.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="nvidia-smi resource check failed"):
        with workflow.resources("eval", "0"):
            pass


def test_cuda_fallback_preserves_memory_occupancy_limit(monkeypatch):
    payload = {"torch": "2.2.2", "cuda": "12.1", "devices": [
        {"visible_index": 0, "name": "NVIDIA RTX A6000", "memory_used_mib": 1024.1,
         "memory_total_mib": 48640},
    ]}
    monkeypatch.setattr(workflow.subprocess, "run",
                        lambda *args, **kwargs: completed(stdout=json.dumps(payload)))
    with pytest.raises(RuntimeError, match="Selected GPU is occupied"):
        workflow._server_cuda_resource_fallback("1", "Driver/library version mismatch")
