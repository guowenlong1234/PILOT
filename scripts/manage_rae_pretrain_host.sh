#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-status}
CONTAINER=gwl-etpr1-rae
ETPNAV_CONTAINER=gwl-etpnav
REPO_ROOT=/home/a6000/gwl/ETP-R1
CONDA_SH=/home/a6000/gwl/miniconda3/etc/profile.d/conda.sh

case "$ACTION" in
    start|resume|status|logs|tail|attach|stop) ;;
    *)
        echo "Usage: $0 {start|resume|status|logs|tail|attach|stop}" >&2
        exit 2
        ;;
esac

ip -br addr | grep -Eq '^eno1[[:space:]]+UP[[:space:]]+10\.10\.10\.2/24' || {
    echo "This script must run on the 4090 host with eno1=10.10.10.2" >&2
    exit 1
}
docker inspect "$CONTAINER" >/dev/null

if [ "$ACTION" = start ] || [ "$ACTION" = resume ]; then
    etpnav_processes=$(docker exec "$ETPNAV_CONTAINER" bash -lc \
        "ps -eo pid,ppid,stat,etime,cmd | grep -E 'torchrun|run.py|train.py' | grep -v grep || true")
    if [ -n "$etpnav_processes" ]; then
        echo "ETPNav still has a training/evaluation process; launch refused:" >&2
        echo "$etpnav_processes" >&2
        exit 1
    fi
    gpu_used=$(nvidia-smi --query-gpu=memory.used \
        --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [ "$gpu_used" -gt 1024 ]; then
        echo "GPU already uses ${gpu_used} MiB; launch refused." >&2
        exit 1
    fi
fi

docker_env_args=()
for variable in \
    ETPR1_PRETRAIN_TMUX_SESSION \
    ETPR1_PRETRAIN_OUTPUT_DIR \
    ETPR1_PRETRAIN_MASTER_PORT \
    ETPR1_PRETRAIN_TRAIN_BATCH_SIZE \
    ETPR1_PRETRAIN_GRADIENT_ACCUMULATION_STEPS \
    ETPR1_PRETRAIN_N_WORKERS \
    ETPR1_PRETRAIN_DISABLE_THREAD_PREFETCH \
    ETPR1_PRETRAIN_ALLOW_WORLD_SIZE_CHANGE \
    ETPR1_PRETRAIN_ALLOW_MODEL_CONFIG_PATH_CHANGE; do
    if [ -n "${!variable:-}" ]; then
        docker_env_args+=(-e "${variable}=${!variable}")
    fi
done

container_command="source ${CONDA_SH} && conda activate etpr1_rae && cd ${REPO_ROOT} && scripts/manage_rae_pretrain.sh ${ACTION}"
if [ "$ACTION" = attach ]; then
    exec docker exec "${docker_env_args[@]}" -it \
        "$CONTAINER" bash -lc "$container_command"
else
    exec docker exec "${docker_env_args[@]}" \
        "$CONTAINER" bash -lc "$container_command"
fi
