#!/bin/bash
# Baseline GPT on OpenWebText. One training run (no sweeps/loops inside).
# Defaults follow launcher.sh's GPT_1kctx_10k_lr30e-4 (nGPT-paper 0.5B reproduce).
# Env-var knobs (with defaults) so a sweep can parameterize it without edits.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch>=2.6 weights_only default rejects the pickled ckpt

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/openwebtext}"   # holds train.bin/val.bin (run data/openwebtext/prepare.py first)
RUN_NAME="${RUN_NAME:-gpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/openwebtext/${RUN_NAME}}"   # ckpt.pt/stat/args/finished + logs

# --- run shape (launcher.sh: 1k ctx, 10k iters, lr 30e-4, 8 GPUs) --------
DEVICES="${DEVICES:-8}"                  # nproc_per_node (GPUs on this node; launcher uses 8)
PER_GPU_BS="${PER_GPU_BS:-8}"            # micro-batch per GPU (= batch_size)
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"      # global batch in SEQUENCES (grad_accum * per_gpu_bs) = 0.5M tok/iter @1k ctx
MAX_ITERS="${MAX_ITERS:-10000}"
LR="${LR:-3e-3}"                         # == launcher.sh 30e-4
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
EVAL_ITERS="${EVAL_ITERS:-1000}"
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-103700}"     # wall-clock stop (for chunked SLURM resume)
MAX_ITERS_PER_LAUNCH="${MAX_ITERS_PER_LAUNCH:-14000}"  # per-job step cap (for chunked SLURM resume)
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"   # bfloat16 | float32 (storage of matrix weights)
COMPILE="${COMPILE:-False}"

# --- model (launcher.sh: nGPT-paper 0.5B) --------------------------------
N_LAYER="${N_LAYER:-24}"
N_HEAD="${N_HEAD:-16}"
N_EMBD="${N_EMBD:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
# 4k-ctx variant (launcher GPT_4kctx_10k_lr30e-4):
#   BLOCK_SIZE=4096 PER_GPU_BS=2 GLOBAL_BATCH=512 MAX_ITERS_PER_LAUNCH=18000 bash scripts/train/owt/gpt.sh

# --- wandb (optional) ----------------------------------------------------
WANDB_LOG="${WANDB_LOG:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-openwebtext-ngpt}"

# grad accumulation is the GLOBAL value; train.py divides it by world size.
GRAD_ACCUM=$(( GLOBAL_BATCH / PER_GPU_BS ))
if (( GRAD_ACCUM * PER_GPU_BS != GLOBAL_BATCH )); then
    echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} not divisible by PER_GPU_BS=${PER_GPU_BS}" >&2; exit 1
fi
if (( GRAD_ACCUM % DEVICES != 0 )); then
    echo "ERROR: grad_accum=${GRAD_ACCUM} not divisible by DEVICES=${DEVICES}" >&2; exit 1
fi

mkdir -p "${OUTPUT_DIR}"
if [ -f "${OUTPUT_DIR}/finished" ]; then echo "${RUN_NAME} already finished"; exit 0; fi
if [ -f "${OUTPUT_DIR}/ckpt.pt" ]; then INIT=resume; else INIT=scratch; fi

cd "${REPO_ROOT}"
torchrun --nnodes=1 --nproc_per_node="${DEVICES}" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py \
    --init_from="${INIT}" \
    --use_nGPT=0 \
    --weight_decay=0.1 --warmup_iters=2000 \
    --learning_rate="${LR}" --min_lr=0.0 \
    --n_layer="${N_LAYER}" --n_head="${N_HEAD}" --n_embd="${N_EMBD}" --block_size="${BLOCK_SIZE}" \
    --weight_dtype="${WEIGHT_DTYPE}" --dtype=bfloat16 \
    --batch_size="${PER_GPU_BS}" --gradient_accumulation_steps="${GRAD_ACCUM}" \
    --max_iters="${MAX_ITERS}" --lr_decay_iters="${MAX_ITERS}" \
    --eval_interval="${EVAL_INTERVAL}" --eval_iters="${EVAL_ITERS}" --log_interval=10 \
    --time_limit_seconds="${TIME_LIMIT_SECONDS}" --max_iters_per_launch="${MAX_ITERS_PER_LAUNCH}" \
    --compile="${COMPILE}" \
    --dataset=openwebtext --data_dir="${DATA_DIR}" --out_dir="${OUTPUT_DIR}" \
    --wandb_log="${WANDB_LOG}" --wandb_project="${WANDB_PROJECT}" --wandb_run_name="${RUN_NAME}"
