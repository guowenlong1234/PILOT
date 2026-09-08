import math,json,torch,numpy as np
from vlnce_baselines.nwm.low_level_context import LowLevelContextEventBuffer
from vlnce_baselines.nwm.etp_adapter import NwmEtpAdapter,RaeGhostInputRequest
events=LowLevelContextEventBuffer()
events.reset_trace()
for i in range(4):
 events.append_observation({"rgb":np.zeros((4,4,3),dtype=np.uint8)},[0,0,-0.25*i],0.0,movement_frame=i>0)
accepted=events.append_observation({"rgb":np.zeros((4,4,3),dtype=np.uint8)},[0,0,-0.75],math.pi/2,movement_frame=True)
events.materialize_pending(render_rgb=None)
payload=events.pop_payload()
a=NwmEtpAdapter();a.reset(1)
for e in payload["events"]:
 if e["type"]=="frame": a.update_context(0,None,e["position"],e["yaw"],latent=torch.zeros(257,768))
frames=a.buffers[0].get_context()
target=[0,0,-1.75]
request=RaeGhostInputRequest(env_index=0,ghost_vp="g",current_position=[0,0,-0.75],current_yaw=math.pi/2,ghost_position=target)
batch=a.build_raenwm_batch([request],device="cpu")
correct=a.build_ghost_condition(env_index=0,ghost_vp="g",current_position=frames[-1].position,current_yaw=frames[-1].yaw,ghost_position=target)
record=batch.records[0]
print(json.dumps({"collision_frame_accepted":accepted,"context_frames":len(frames),"context_last_yaw_deg":math.degrees(frames[-1].yaw),"current_yaw_deg":90,"skipped":batch.skipped,"actual_condition":{"forward_m":record.local_dx_m,"left_m":record.local_dy_m,"dtheta":record.condition.dtheta},"condition_from_context_last":{"forward_m":correct.local_dx_m,"left_m":correct.local_dy_m,"dtheta":correct.condition.dtheta}}))
assert not accepted and len(frames)==4 and abs(record.local_dy_m+1)<1e-6 and abs(correct.local_dx_m-1)<1e-6
print("REPRODUCED: last-frame source and query source disagree; no guard")

