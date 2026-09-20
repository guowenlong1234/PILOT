"""Verify live-row scoring exactly reproduces selected offline CUDA/BF16 inference."""
import argparse
import json
from pathlib import Path
import torch
from evaluate_stage2_e24 import predict, collate
from vlnce_baselines.nwm.active_lookahead.stage2_online import score_rows
from vlnce_baselines.nwm.active_lookahead.stage2_data import load, atomic_json
from vlnce_baselines.nwm.active_lookahead.residual_head import InterleavedCrossModalTopKFutureLogitResidualHead as Head

p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--head',required=True);p.add_argument('--report',required=True);a=p.parse_args()
root=Path(a.data);m=json.loads((root/'dataset_manifest.json').read_text());rows=[]
for entry in m['entries'][:4]:
 ep=load(root/entry['file'])
 rows.extend(dict(r,text_tokens=ep['text_tokens'],episode_id=ep['episode_id'],scene_id=ep['scene_id']) for r in ep['rows'])
ck=load(a.head);head=Head(**ck['model_config']).cuda().eval();head.load_state_dict(ck['future_head_state_dict'],strict=True)
with torch.inference_mode():
 for start in range(0,len(rows),16):
  part=rows[start:start+16];b=collate(part);b={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()}
  with torch.autocast('cuda',dtype=torch.bfloat16): expected=predict(head,b)
  expected=(1.5*expected.float()).clamp(-1,1).masked_fill(~b['topk_valid_mask'],0)
  # Remove targets entirely: inference cannot depend on teacher labels.
  no_labels=[{k:v for k,v in r.items() if not k.startswith('teacher_')} for r in part]
  stream=torch.cuda.Stream(); current=torch.cuda.current_stream(); stream.wait_stream(current)
  with torch.cuda.stream(stream): actual=score_rows(head,no_labels,'cuda',1.5)
  current.wait_stream(stream); actual.record_stream(current)
  assert torch.equal(actual,expected),float((actual-expected).abs().max())
  assert torch.count_nonzero(score_rows(head,no_labels,'cuda',0))==0
atomic_json(a.report,dict(status='passed',rows=len(rows),bitwise_equal=True,teacher_fields_removed=True,zero_gain_exact=True))
print(json.dumps(dict(status='passed',rows=len(rows))))
