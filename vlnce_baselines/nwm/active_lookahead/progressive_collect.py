"""Full expansion collection; labels are attached only after all prediction."""
import json
from pathlib import Path
import torch
from .stage2_collect import Stage2Collector
from .progressive_prediction import ProgressivePrediction
from .progressive_data import ProgressiveEpisodeWriter
from .progressive_online import live_provenance, validate_progressive_config
from .progressive_training import source_contract


class ProgressiveCollector(Stage2Collector):
    def __init__(self,trainer):
        pcfg=trainer.config.MODEL.PROGRESSIVE
        validate_progressive_config(pcfg)
        # Initialize frozen runtime only, with no legacy v1 writer/provenance.
        super().__init__(trainer,prediction_only=True)
        self.cfg=trainer.config.MODEL.STAGE2_COLLECT
        self.prediction_only=False
        p=json.loads(Path(self.cfg.provenance).read_text())
        actual=live_provenance(trainer,pcfg.max_future_depth,pcfg.distance_scale)
        if source_contract(p)!=source_contract(actual) or p['split']!=actual['split'] or p['behavior']!='stage1_argmax':
            raise ValueError('collection source contract differs from live assets')
        self.writer=ProgressiveEpisodeWriter(self.cfg.output,p)
        self.predictor=ProgressivePrediction(self,pcfg.max_future_depth,pcfg.distance_scale)

    @torch.no_grad()
    def predict_step(self,nav_inputs,nav_outs,text,text_mask,no_vp_left,step):
        decision=self.predictor.initialize_decision(nav_inputs,nav_outs,text,text_mask,no_vp_left,step)
        for _ in range(self.predictor.max_future_depth):
            self.predictor.predict_next_depth(decision)
        return self.predictor.finalize_decision(decision)
