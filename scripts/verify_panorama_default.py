#!/usr/bin/env python3
"""Exercise real default navigation plumbing without a full performance eval."""
import argparse
import json
import math
from pathlib import Path
import runpy
import sys

import numpy as np


class Done(Exception):pass


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--environments',type=int,default=2)
    args=p.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    from vlnce_baselines.nwm.runtime import NwmPredictionRuntime
    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer
    from rgb_only_optimization import common
    old=NwmPredictionRuntime.predict;old_rollout=RLTrainer.rollout;trainer=[None]
    report={'prediction_calls':0,'nonempty_calls':0,'queries':0,'reverse_queries':0,'mode':None}
    def rollout(self,*a,**kw):trainer[0]=self;return old_rollout(self,*a,**kw)
    def predict(self,queries,**kw):
        assert self.panorama_mode=='world_exact_select','default did not select panorama'
        result=old(self,queries,**kw);report['prediction_calls']+=1
        report['mode']=self.panorama_mode
        rows=result.meta.get('records',[])
        if rows:
            report['nonempty_calls']+=1;report['queries']+=len(rows)
            report['reverse_queries']+=self.panorama_predictor.last_diagnostics['reverse_queries']
            lookup={(q.env_index,q.query_id):q for q in queries}
            for record,source in zip(rows,result.meta['sources']):
                q=lookup[(record.env_index,record.query_id)]
                np.testing.assert_allclose(source['target_position'],q.target_position,atol=1e-6)
                d=np.asarray(q.target_position)-q.current_position
                heading=math.atan2(-d[0],-d[2])
                assert abs((source['target_yaw']-heading+math.pi)%(2*math.pi)-math.pi)<1e-5
                assert abs(record.condition.dtheta)<1e-6
            print('DEFAULT_PANORAMA '+json.dumps(report),flush=True)
        if report['nonempty_calls']>=3:raise Done()
        return result
    NwmPredictionRuntime.predict=predict;RLTrainer.rollout=rollout
    opts=common('0',args.environments)
    opts.update({'IL.freeze_navigation_backbone':False,'IL.lr':2e-6,'IL.rgb_fusion_lr':1e-5,
        'EVAL.CKPT_PATH_DIR':str(Path(args.checkpoint).resolve()),'EVAL.EPISODE_COUNT':args.environments,
        'EVAL.SAVE_RESULTS':False,'EVAL.USE_CKPT_CONFIG':False,
        'CHECKPOINT_FOLDER':str(root/'checkpoints')+'/', 'TENSORBOARD_DIR':str(root/'tb')+'/',
        'RESULTS_DIR':str(root/'results')+'/'})
    sys.argv=['run.py','--exp_name','panorama_default_smoke','--run-type','eval','--exp-config','run_r2r/iter_train_rae_dino_ghost_concat.yaml']
    for key,value in opts.items():sys.argv.extend([key,str(value)])
    report['command']=list(sys.argv)
    try:
        try:runpy.run_path('run.py',run_name='__main__')
        except Done:pass
        assert report['nonempty_calls']>=3
        report['passed']=True
    finally:
        if trainer[0] is not None:trainer[0].envs.close()
        NwmPredictionRuntime.predict=old;RLTrainer.rollout=old_rollout
        (root/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
