"""Exercise the real audit hooks with graph updates, without a GPU simulator."""
import json
import sys

import pytest

import scripts.check_ghost_concat_rollout as audit
from test_ghost_concat_persistent import _graph, _observe, _inputs, _pred, _adapter
from test_ghost_concat_trainer import trainer
from vlnce_baselines.nwm.ghost_concat_fusion import apply_ghost_concat_to_graph


@pytest.mark.parametrize("missing_writeback", [False, True])
def test_rollout_audit_reads_live_graph_and_detects_missing_writeback(tmp_path, monkeypatch, missing_writeback):
    graph = _graph()
    _observe(graph, [[1.] * 4])
    obj = trainer()
    obj.config.MODEL.RAENWM.ghost_concat_memory_mode = "persistent_node_state"
    obj.gmaps = [graph]
    obj.raenwm_rgb_fusion_adapter = _adapter()
    obj.last_candidate_q0_prediction_diagnostics = {}
    obj._accumulate_rgb_fusion_diagnostics = lambda *args: None
    monkeypatch.setattr(audit.ETP, "forward", lambda *args, **kwargs: None)
    if missing_writeback:
        def broken(self, inputs, prediction):
            result, stats = apply_ghost_concat_to_graph(inputs, prediction, self.raenwm_rgb_fusion_adapter)
            self.last_raenwm_rgb_fusion_diagnostics = [stats]
            return result
        monkeypatch.setattr(audit.RLTrainer, "_apply_ghost_concat_prediction", broken)
    def rollout(*args, **kwargs):
        for prediction in (_pred(), None):
            audit.ETP.forward(None, mode="panorama")
            obj._apply_ghost_concat_prediction(_inputs(graph), prediction)
    monkeypatch.setattr(audit.runpy, "run_path", rollout)
    report_path = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--report", str(report_path)])
    if missing_writeback:
        with pytest.raises(AssertionError, match="live ghost state"):
            audit.main()
        assert json.loads(report_path.read_text())["ok"] is False
    else:
        audit.main()
        report = json.loads(report_path.read_text())
        assert report["ok"] and report["cross_step_retention_exercised"]
        assert report["persistent_writeback_count"] == 1
        assert report["cross_step_retained_rows"] == 1
