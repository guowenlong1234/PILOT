#!/usr/bin/env python3
"""Instrument one real SS-ETP-R1 rollout; pass the usual run.py arguments.

Run in the target project runtime with --report <json> and an isolated output
directory. Expensive tensor equality checks are confined to this validation
tool; they are not part of production training/evaluation.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

import torch

from vlnce_baselines.models.R1Policy import ETP
from vlnce_baselines.nwm.ghost_concat_fusion import GhostConcatFusionAdapter
from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--report", required=True)
    own, remaining = parser.parse_known_args()
    report = dict(ok=False, checked_steps=0, panorama_calls=0, mlp_calls=0,
                  max_ghost_batch=0, fused_ghosts=0, changed_input_rows=0)
    pending_pano = [0]
    old_forward = ETP.forward
    old_adapter = GhostConcatFusionAdapter.forward
    old_predict = RLTrainer._run_raenwm_rgb_fusion_prediction
    old_apply = RLTrainer._apply_ghost_concat_prediction

    def forward(self, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else None)
        if mode == "panorama":
            pending_pano[0] += 1
            report["panorama_calls"] += 1
        return old_forward(self, *args, **kwargs)

    def adapter(self, observed, predicted):
        report["mlp_calls"] += 1
        report["max_ghost_batch"] = max(report["max_ghost_batch"], observed.shape[0])
        return old_adapter(self, observed, predicted)

    def predict(self, *args, **kwargs):
        wp = kwargs.get("wp_outputs", args[4] if len(args) > 4 else None)
        before = [row.detach().clone() for row in wp["cand_rgb"]]
        result = old_predict(self, *args, **kwargs)
        assert all(torch.equal(x, y) for x, y in zip(before, wp["cand_rgb"])), "candidate RGB mutated"
        return result

    def apply(self, nav_inputs, prediction):
        assert pending_pano[0] == 1, "expected exactly one panorama forward per graph step"
        pending_pano[0] = 0
        tensors = [(value, value.detach().clone()) for graph in self.gmaps
                   for value in list(graph.node_embeds.values()) + [v[0] for v in graph.ghost_embeds.values()]]
        counts = [dict((key, value[1]) for key, value in graph.ghost_embeds.items()) for graph in self.gmaps]
        original = nav_inputs["gmap_img_fts"].detach().clone()
        before_calls = report["mlp_calls"]
        result = old_apply(self, nav_inputs, prediction)
        assert report["mlp_calls"] - before_calls <= 1, "more than one MLP batch in graph step"
        assert all(torch.equal(value, copy) for value, copy in tensors), "persistent graph modified"
        assert counts == [dict((key, value[1]) for key, value in graph.ghost_embeds.items()) for graph in self.gmaps]
        assert torch.equal(original, nav_inputs["gmap_img_fts"]), "raw navigation inputs modified"
        report["changed_input_rows"] += int((result["gmap_img_fts"] != original).any(-1).sum().item())
        report["fused_ghosts"] += int(self.last_raenwm_rgb_fusion_diagnostics[0]["fused_candidate_count"])
        report["checked_steps"] += 1
        return result

    ETP.forward = forward
    GhostConcatFusionAdapter.forward = adapter
    RLTrainer._run_raenwm_rgb_fusion_prediction = predict
    RLTrainer._apply_ghost_concat_prediction = apply
    try:
        sys.argv = ["run.py"] + remaining
        runpy.run_path("run.py", run_name="__main__")
        assert report["checked_steps"] > 0 and report["fused_ghosts"] > 0
        assert report["changed_input_rows"] > 0, "validation checkpoint did not exercise a learned residual"
        report["ok"] = True
    finally:
        ETP.forward = old_forward
        GhostConcatFusionAdapter.forward = old_adapter
        RLTrainer._run_raenwm_rgb_fusion_prediction = old_predict
        RLTrainer._apply_ghost_concat_prediction = old_apply
        path = Path(own.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
