"""Versioned frozen RAE feature and progressive geometry contract (stdlib only)."""
FORMAT = 'etpr1-progressive-episode-v1'
SPACE = 'raw_cls+normalized_patch_fp16'
# axis 0 follows increasing local waypoint angle; axis 1 is forward.
GEOMETRY = 'q0_local_angle_forward_sinyaw_cosyaw_path_v1'
FUTURE_TOKEN_COUNT = 257
FUTURE_FEATURE_DIM = 768
FEATURE_CONTRACT = dict(token_count=FUTURE_TOKEN_COUNT,feature_dim=FUTURE_FEATURE_DIM,
                        owner_dim=FUTURE_FEATURE_DIM,text_dim=FUTURE_FEATURE_DIM)
PREDICTION_CONTRACT = dict(seed=20260916,context_size=4,num_steps=10,
    final_only_euler=False,max_horizon=64.,condition_source_pose='context_last',
    panorama_context_mode='world_exact_select',none_threshold=.3,
    request_batch_size=1,score_scale='logit')
