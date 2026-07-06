#!/bin/bash
# nGPT (normalized transformer on the hypersphere) on OpenWebText. One run.
# Defaults follow launcher.sh's nGPT_1kctx_10k_lr30e-4 (nGPT-paper 0.5B reproduce).
# Config is composed by Hydra (configs/); knobs below become hydra overrides.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch>=2.6 weights_only default rejects the pickled ckpt

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/openwebtext}"   # holds train.bin/val.bin (run data/openwebtext/prepare.py first)
RUN_NAME="${RUN_NAME:-ngpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/openwebtext/${RUN_NAME}}"   # checkpoints/, eval/, .hydra/, stat, args

# --- run shape (launcher.sh: 1k ctx, 10k iters, lr 30e-4, 8 GPUs) --------
DEVICES="${DEVICES:-8}"                  # nproc_per_node (GPUs on this node; launcher uses 8)
BATCH_SIZE="${BATCH_SIZE:-8}"            # per-GPU micro-batch (= train.py batch_size)
GRAD_ACCUM="${GRAD_ACCUM:-64}"          # gradient accumulation (global batch = GRAD_ACCUM * BATCH_SIZE = 512 seqs @1k ctx)
MAX_ITERS="${MAX_ITERS:-10000}"
LR="${LR:-3e-3}"                         # == launcher.sh 30e-4
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
EVAL_ITERS="${EVAL_ITERS:-1000}"
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-103700}"     # wall-clock stop (for chunked SLURM resume)
MAX_ITERS_PER_LAUNCH="${MAX_ITERS_PER_LAUNCH:-14000}"  # per-job step cap (for chunked SLURM resume)
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"   # bfloat16 | float32 (storage of matrix weights)
COMPILE="${COMPILE:-false}"                # hydra bool: lowercase true|false

# model architecture (n_layer/n_head/n_embd/block_size) comes from configs/model/gpt2_medium.yaml
# 4k-ctx variant: use a model config with block_size=4096, plus BATCH_SIZE=2 GRAD_ACCUM=256 MAX_ITERS_PER_LAUNCH=18000

# --- wandb (optional) ----------------------------------------------------
WANDB_LOG="${WANDB_LOG:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-openwebtext-ngpt}"

cd "${REPO_ROOT}"
# Hydra overrides (no leading --). model=gpt2_medium (0.5B arch) + optimizer=ngpt_opt + use_nGPT=1 = nGPT recipe;
# train.py derives base_scale=1/sqrt(n_embd). out_dir prep + resume/finished detection now in train.py (utils.Initialization).
torchrun --nnodes=1 --nproc_per_node="${DEVICES}" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py \
    model=gpt2_medium \
    optimizer=ngpt_opt \
    use_nGPT=1 \
    data=openwebtext \
    learning_rate="${LR}" min_lr=0.0 \
    weight_dtype="${WEIGHT_DTYPE}" dtype=bfloat16 \
    batch_size="${BATCH_SIZE}" gradient_accumulation_steps="${GRAD_ACCUM}" \
    max_iters="${MAX_ITERS}" lr_decay_iters="${MAX_ITERS}" \
    eval_interval="${EVAL_INTERVAL}" eval_iters="${EVAL_ITERS}" log_interval=10 \
    time_limit_seconds="${TIME_LIMIT_SECONDS}" max_iters_per_launch="${MAX_ITERS_PER_LAUNCH}" \
    compile="${COMPILE}" \
    data_dir="${DATA_DIR}" out_dir="${OUTPUT_DIR}" \
    wandb_log="${WANDB_LOG}" wandb_project="${WANDB_PROJECT}" wandb_run_name="${RUN_NAME}"
