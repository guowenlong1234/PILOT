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
import weakref

import torch

from vlnce_baselines.models.R1Policy import ETP
from vlnce_baselines.nwm.ghost_concat_fusion import GhostConcatFusionAdapter
from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--report", required=True)
    own, remaining = parser.parse_known_args()
    report = dict(ok=False, checked_steps=0, panorama_calls=0, mlp_calls=0,
                  max_ghost_batch=0, fused_ghosts=0, changed_input_rows=0,
                  memory_mode=None, persistent_writeback_count=0,
                  persistent_retained_state_count=0, persistent_state_norm_sum=0.0,
                  persistent_state_norm_count=0, graph_state_rows_checked=0,
                  no_prediction_retained_rows=0, cross_step_retained_rows=0)
    previous_states = weakref.WeakKeyDictionary()
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
        memory_mode = self._ghost_concat_memory_mode()
        assert report["memory_mode"] in (None, memory_mode), "memory mode changed during rollout"
        report["memory_mode"] = memory_mode
        persistent = memory_mode == "persistent_node_state"
        # Snapshot values by key, then read the dictionaries again after apply:
        # checking only old tensor references misses replacement-based writes.
        nodes = [{key: value.detach().clone() for key, value in graph.node_embeds.items()}
                 for graph in self.gmaps]
        ghosts = [{key: (value[0].detach().clone(), value[1])
                   for key, value in graph.ghost_embeds.items()} for graph in self.gmaps]
        counts = [{key: value[1] for key, value in graph.ghost_embeds.items()} for graph in self.gmaps]
        state_vps = [set(graph.ghost_concat_state_vps) for graph in self.gmaps]
        predicted_keys = {(int(record.env_index), str(record.ghost_vp))
                          for record in ((getattr(prediction, "meta", None) or {}).get("records", []))}
        if getattr(prediction, "pred_cls_raw", None) is None:
            predicted_keys = set()
        original = nav_inputs["gmap_img_fts"].detach().clone()
        before_calls = report["mlp_calls"]
        result = old_apply(self, nav_inputs, prediction)
        assert report["mlp_calls"] - before_calls <= 1, "more than one MLP batch in graph step"
        for env, graph in enumerate(self.gmaps):
            assert nodes[env].keys() == graph.node_embeds.keys(), "visited nodes replaced"
            assert all(torch.equal(graph.node_embeds[key], value)
                       for key, value in nodes[env].items()), "visited actual observation changed"
            assert counts[env] == {key: value[1] for key, value in graph.ghost_embeds.items()}, "prediction incremented observation count"
            for vp, (before, count) in ghosts[env].items():
                current = graph.ghost_embeds[vp][0]
                if not persistent or (env, str(vp)) not in predicted_keys:
                    assert torch.equal(current, before), "unpredicted/legacy ghost state changed"
                if persistent and vp in state_vps[env] and (env, str(vp)) not in predicted_keys:
                    report["no_prediction_retained_rows"] += 1
                    previous = previous_states.get(graph, {}).get(vp)
                    # New real observations may legitimately update the state.
                    # With unchanged observation count, the preceding step's
                    # fused memory must survive unchanged through this step.
                    if previous is not None and previous[1] == count:
                        assert torch.equal(current, previous[0]), "memory lost between graph steps"
                        report["cross_step_retained_rows"] += 1
            if persistent:
                for col, vp in enumerate(nav_inputs["gmap_vp_ids"][env]):
                    if vp not in graph.ghost_embeds:
                        continue
                    # Accumulator/count may introduce one rounding operation.
                    torch.testing.assert_close(
                        graph.get_node_embeds(vp).detach(),
                        result["gmap_img_fts"][env, col].detach(),
                        rtol=1e-5, atol=1e-6,
                        msg="live ghost state differs from the fused navigation input",
                    )
                    report["graph_state_rows_checked"] += 1
                previous_states[graph] = {
                    vp: (graph.ghost_embeds[vp][0].detach().clone(), graph.ghost_embeds[vp][1])
                    for vp in graph.ghost_concat_state_vps
                }
        assert torch.equal(original, nav_inputs["gmap_img_fts"]), "raw navigation inputs modified"
        visited = nav_inputs["gmap_visited_masks"].bool()
        assert torch.equal(result["gmap_img_fts"][visited], original[visited]), "visited navigation rows changed"
        # The graph tensor check above catches a missing writeback even if the
        # diagnostics incorrectly claim success; counts/norms expose coverage.
        diagnostics = self.last_raenwm_rgb_fusion_diagnostics[0]
        for key in ("persistent_writeback_count", "persistent_retained_state_count",
                    "persistent_state_norm_sum", "persistent_state_norm_count"):
            report[key] += float(diagnostics.get(key, 0))
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
        if report["memory_mode"] == "persistent_node_state":
            assert report["persistent_writeback_count"] > 0, "no persistent writeback exercised"
            assert report["graph_state_rows_checked"] > 0, "no live graph state checked"
        report["ok"] = True
    finally:
        ETP.forward = old_forward
        GhostConcatFusionAdapter.forward = old_adapter
        RLTrainer._run_raenwm_rgb_fusion_prediction = old_predict
        RLTrainer._apply_ghost_concat_prediction = old_apply
        count = report["persistent_state_norm_count"]
        report["persistent_state_norm_mean"] = report["persistent_state_norm_sum"] / count if count else 0.0
        report["cross_step_retention_exercised"] = report["cross_step_retained_rows"] > 0
        path = Path(own.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
