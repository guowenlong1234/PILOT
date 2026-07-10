#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${ETPR1_RUNTIME_ACTIVE:-}" != "1" ]]; then
  echo "ERROR: run through scripts/etpr1_rae_runtime_exec.sh" >&2
  exit 2
fi

source_commit="${ETPR1_SOURCE_COMMIT:-}"
if [[ -z "$source_commit" ]]; then
  source_commit="$(git rev-parse --short HEAD 2>/dev/null || true)"
fi
source_commit="${source_commit:-nogit}"
source_id="$(printf '%s' "$source_commit" | tr -cs 'A-Za-z0-9._-' '_')"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${source_id}_${timestamp}_$$"
run_root="${ETPR1_RAE_SMOKE_ROOT:-data/logs/rae_dino_smoke/$run_id}"
pretrain_root="pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp_smoke_$run_id"
pretrain_config="$run_root/fixtures/pretrain_smoke_config.json"
pretrain_initial_projection="$run_root/fixtures/pretrain_initial_projection.pt"

if [[ -e "$run_root" || -e "$pretrain_root" ]]; then
  echo "ERROR: smoke output already exists" >&2
  echo "run_root=$run_root" >&2
  echo "pretrain_root=$pretrain_root" >&2
  exit 2
fi

mkdir -p "$run_root/logs" "$run_root/matplotlib"
export MPLCONFIGDIR="$PROJECT_ROOT/$run_root/matplotlib"

run_stage() {
  local name="$1"
  local seconds="$2"
  shift 2
  local log="$run_root/logs/$name.log"
  {
    printf 'RUN_ID=%s\n' "$run_id"
    printf 'SOURCE_COMMIT=%s\n' "$source_commit"
    printf 'COMMAND='
    printf ' %q' "$@"
    printf '\n'
  } > "$log"
  if timeout --signal=TERM --kill-after=30s "${seconds}s" "$@" >> "$log" 2>&1; then
    echo "PASS $name log=$log"
  else
    local status=$?
    echo "FAIL $name status=$status log=$log" >&2
    tail -n 80 "$log" >&2
    exit "$status"
  fi
}

run_gpu_stage() {
  local name="$1"
  local seconds="$2"
  shift 2
  local gpu_log="$run_root/logs/${name}_gpu_before.log"
  nvidia-smi > "$gpu_log"
  mapfile -t compute_pids < <(
    nvidia-smi --query-compute-apps=pid \
      --format=csv,noheader,nounits \
      | sed '/^[[:space:]]*$/d'
  )
  if (( ${#compute_pids[@]} > 0 )); then
    printf 'REFUSE_EXISTING_GPU_COMPUTE_PIDS=%s\n' \
      "${compute_pids[*]}" >> "$gpu_log"
    echo "FAIL $name existing GPU compute PIDs: ${compute_pids[*]}" >&2
    exit 3
  fi
  run_stage "$name" "$seconds" "$@"
}

online_args=(
  SIMULATOR_GPU_IDS "[0]"
  TORCH_GPU_IDS "[0]"
  GPU_NUMBERS 1
  NUM_ENVIRONMENTS 1
)

run_stage runtime 120 python scripts/inspect_etpr1_runtime.py
run_stage contracts 300 pytest -q \
  tests/integration/test_pretrain_to_online_projection.py \
  tests/integration/test_training_stage_freeze.py
run_gpu_stage test_real_mlm_batch 300 pytest -q \
  tests/integration/test_pretrain_tasks_smoke.py \
  -k test_real_mlm_batch
run_gpu_stage test_real_sap_batch 300 pytest -q \
  tests/integration/test_pretrain_tasks_smoke.py \
  -k test_real_sap_batch
run_stage prepare_pretrain_fixture 120 \
  python scripts/prepare_rae_smoke_pretrain.py \
  --source-config pretrain_src/run_pt/mix_pretrain_rae_dino.json \
  --output-config "$pretrain_config" \
  --seed 20260710
run_gpu_stage pretrain_one_step 900 \
  env ETPR1_RAE_SMOKE_INITIAL_PROJECTION="$pretrain_initial_projection" \
  torchrun --nproc_per_node=1 --master_port=23401 \
  pretrain_src/pretrain_src/train_r2r.py \
  --world_size 1 --vlnbert cmt \
  --model_config pretrain_src/run_pt/mix_model_config_rae_dino.json \
  --config "$pretrain_config" \
  --output_dir "$pretrain_root" \
  --num_train_steps 1 --train_batch_size 1 --val_batch_size 1 \
  --valid_steps 1000 --log_steps 1 --n_workers 0 \
  --seed 20260710

pretrain_checkpoint="$pretrain_root/ckpts/model_step_1.pt"
run_stage audit_pretrain 120 python scripts/audit_rae_smoke.py \
  pretrain "$pretrain_checkpoint" \
  --initial-projection "$pretrain_initial_projection"

r2r_sft_name="${run_id}_r2r_sft"
r2r_sft_root="$run_root/r2r_sft"
run_gpu_stage r2r_sft 900 python run.py \
  --exp_name "$r2r_sft_name" --run-type dagger \
  --exp-config run_r2r/iter_train_rae_dino.yaml \
  "${online_args[@]}" \
  IL.iters 1 IL.log_every 1 IL.load_from_ckpt False IL.warmup_iters 0 \
  IL.amp_init_scale 16384.0 \
  CHECKPOINT_FOLDER "$r2r_sft_root/checkpoints/" \
  TENSORBOARD_DIR "$r2r_sft_root/tensorboard/" \
  RESULTS_DIR "$r2r_sft_root/results/" \
  MODEL.pretrained_path "$pretrain_checkpoint"
r2r_sft_checkpoint="$r2r_sft_root/checkpoints/$r2r_sft_name/ckpt.iter1.pth"
run_stage audit_r2r_sft 120 python scripts/audit_rae_smoke.py \
  sft "$pretrain_checkpoint" "$r2r_sft_checkpoint" --iteration 1

r2r_grpo_name="${run_id}_r2r_grpo"
r2r_grpo_root="$run_root/r2r_grpo"
run_gpu_stage r2r_grpo 900 python run.py \
  --exp_name "$r2r_grpo_name" --run-type grpo \
  --exp-config run_r2r/iter_train_rae_dino.yaml \
  "${online_args[@]}" \
  TRAINER_NAME GRPO-R1 \
  GRPO.iters 1 GRPO.log_every 1 GRPO.sample_num 2 \
  GRPO.update_epochs 1 GRPO.load_from_ckpt True \
  GRPO.is_requeue False GRPO.warmup_iters 0 \
  GRPO.ckpt_to_load "$r2r_sft_checkpoint" \
  CHECKPOINT_FOLDER "$r2r_grpo_root/checkpoints/" \
  TENSORBOARD_DIR "$r2r_grpo_root/tensorboard/" \
  RESULTS_DIR "$r2r_grpo_root/results/" \
  MODEL.pretrained_path "$pretrain_checkpoint"
r2r_grpo_checkpoint="$r2r_grpo_root/checkpoints/$r2r_grpo_name/ckpt.iter1.pth"
run_stage audit_r2r_grpo 120 python scripts/audit_rae_smoke.py \
  frozen "$r2r_sft_checkpoint" "$r2r_grpo_checkpoint" --iteration 1

r2r_eval_name="${run_id}_r2r_eval"
r2r_eval_root="$run_root/r2r_eval"
run_gpu_stage r2r_eval 900 python run.py \
  --exp_name "$r2r_eval_name" --run-type eval \
  --exp-config run_r2r/iter_train_rae_dino.yaml \
  "${online_args[@]}" \
  EVAL.CKPT_PATH_DIR "$r2r_sft_checkpoint" \
  EVAL.EPISODE_COUNT 1 EVAL.SAVE_RESULTS False \
  CHECKPOINT_FOLDER "$r2r_eval_root/checkpoints/" \
  TENSORBOARD_DIR "$r2r_eval_root/tensorboard/" \
  RESULTS_DIR "$r2r_eval_root/results/" \
  MODEL.pretrained_path "$pretrain_checkpoint"

rxr_sft_name="${run_id}_rxr_sft"
rxr_sft_root="$run_root/rxr_sft"
run_gpu_stage rxr_sft 900 python run.py \
  --exp_name "$rxr_sft_name" --run-type dagger \
  --exp-config run_rxr/iter_train_rae_dino.yaml \
  "${online_args[@]}" \
  IL.iters 1 IL.log_every 1 IL.load_from_ckpt False IL.warmup_iters 0 \
  IL.amp_init_scale 16384.0 \
  CHECKPOINT_FOLDER "$rxr_sft_root/checkpoints/" \
  TENSORBOARD_DIR "$rxr_sft_root/tensorboard/" \
  RESULTS_DIR "$rxr_sft_root/results/" \
  MODEL.pretrained_path "$pretrain_checkpoint"
rxr_sft_checkpoint="$rxr_sft_root/checkpoints/$rxr_sft_name/ckpt.iter1.pth"
run_stage audit_rxr_sft 120 python scripts/audit_rae_smoke.py \
  sft "$pretrain_checkpoint" "$rxr_sft_checkpoint" --iteration 1

rxr_eval_name="${run_id}_rxr_eval"
rxr_eval_root="$run_root/rxr_eval"
run_gpu_stage rxr_eval 900 python run.py \
  --exp_name "$rxr_eval_name" --run-type eval \
  --exp-config run_rxr/iter_train_rae_dino.yaml \
  "${online_args[@]}" \
  EVAL.CKPT_PATH_DIR "$rxr_sft_checkpoint" \
  EVAL.EPISODE_COUNT 1 EVAL.SAVE_RESULTS False \
  CHECKPOINT_FOLDER "$rxr_eval_root/checkpoints/" \
  TENSORBOARD_DIR "$rxr_eval_root/tensorboard/" \
  RESULTS_DIR "$rxr_eval_root/results/" \
  MODEL.pretrained_path "$pretrain_checkpoint"

echo "ALL_PASS run_id=$run_id run_root=$run_root"
