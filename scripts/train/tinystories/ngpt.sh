#!/bin/bash
# nGPT (normalized transformer on the hypersphere) on TinyStories. One run.
# Config is composed by Hydra (configs/); knobs below become hydra overrides.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch>=2.6 weights_only default rejects the pickled ckpt

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/tinystories}"   # holds train.bin/val.bin (run data/tinystories/prepare.py first)
RUN_NAME="${RUN_NAME:-ngpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/tinystories/${RUN_NAME}}"   # checkpoints/, eval/, .hydra/, stat, args

# --- run shape -----------------------------------------------------------
DEVICES="${DEVICES:-1}"                 # nproc_per_node (GPUs on this node)
BATCH_SIZE="${BATCH_SIZE:-8}"           # per-GPU micro-batch (= train.py batch_size)
GRAD_ACCUM="${GRAD_ACCUM:-64}"          # gradient accumulation (global batch = GRAD_ACCUM * BATCH_SIZE = 512 seqs)
MAX_ITERS="${MAX_ITERS:-30000}"
LR="${LR:-3e-3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"   # bfloat16 | float32 (storage of matrix weights)
COMPILE="${COMPILE:-false}"                # hydra bool: lowercase true|false

# --- wandb (optional) ----------------------------------------------------
WANDB_LOG="${WANDB_LOG:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-tinystories-ngpt}"

cd "${REPO_ROOT}"
# Hydra overrides (no leading --). model=small (size) + optimizer=ngpt_opt + use_nGPT=1 = nGPT recipe;
# train.py derives base_scale=1/sqrt(n_embd). out_dir prep + resume/finished detection now in train.py (utils.Initialization).
torchrun --nnodes=1 --nproc_per_node="${DEVICES}" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py \
    model=small \
    optimizer=ngpt_opt \
    use_nGPT=1 \
    data=tinystories \
    learning_rate="${LR}" min_lr=0.0 \
    weight_dtype="${WEIGHT_DTYPE}" dtype=bfloat16 \
    batch_size="${BATCH_SIZE}" gradient_accumulation_steps="${GRAD_ACCUM}" \
    max_iters="${MAX_ITERS}" lr_decay_iters="${MAX_ITERS}" \
    eval_interval="${EVAL_INTERVAL}" eval_iters="${EVAL_ITERS}" log_interval=10 \
    compile="${COMPILE}" \
    data_dir="${DATA_DIR}" out_dir="${OUTPUT_DIR}" \
    wandb_log="${WANDB_LOG}" wandb_project="${WANDB_PROJECT}" wandb_run_name="${RUN_NAME}"
