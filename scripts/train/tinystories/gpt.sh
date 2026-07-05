#!/bin/bash
# Baseline GPT on TinyStories. One training run (no sweeps/loops inside).
# Config is composed by Hydra (configs/); knobs below become hydra overrides.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch>=2.6 weights_only default rejects the pickled ckpt

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/tinystories}"   # holds train.bin/val.bin (run data/tinystories/prepare.py first)
RUN_NAME="${RUN_NAME:-gpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/tinystories/${RUN_NAME}}"   # checkpoints/, eval/, .hydra/, stat, args

# --- run shape -----------------------------------------------------------
DEVICES="${DEVICES:-1}"                 # nproc_per_node (GPUs on this node)
PER_GPU_BS="${PER_GPU_BS:-8}"           # micro-batch per GPU (= batch_size)
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"     # global batch in SEQUENCES (grad_accum * per_gpu_bs)
MAX_ITERS="${MAX_ITERS:-30000}"
LR="${LR:-3e-3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"   # bfloat16 | float32 (storage of matrix weights)
COMPILE="${COMPILE:-false}"                # hydra bool: lowercase true|false

# --- model (s-flm "small": GPT-2-small scale) ----------------------------
N_LAYER="${N_LAYER:-12}"
N_HEAD="${N_HEAD:-12}"
N_EMBD="${N_EMBD:-768}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"

# --- wandb (optional) ----------------------------------------------------
WANDB_LOG="${WANDB_LOG:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-tinystories-ngpt}"

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
if [ -f "${OUTPUT_DIR}/checkpoints/ckpt.pt" ]; then INIT=resume; else INIT=scratch; fi

cd "${REPO_ROOT}"
# Hydra overrides (no leading --). model=small (size) + optimizer=gpt_opt + use_nGPT=0 = GPT recipe.
torchrun --nnodes=1 --nproc_per_node="${DEVICES}" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py \
    model=small \
    optimizer=gpt_opt \
    use_nGPT=0 \
    data=tinystories \
    init_from="${INIT}" \
    learning_rate="${LR}" min_lr=0.0 \
    n_layer="${N_LAYER}" n_head="${N_HEAD}" n_embd="${N_EMBD}" block_size="${BLOCK_SIZE}" \
    weight_dtype="${WEIGHT_DTYPE}" dtype=bfloat16 \
    batch_size="${PER_GPU_BS}" gradient_accumulation_steps="${GRAD_ACCUM}" \
    max_iters="${MAX_ITERS}" lr_decay_iters="${MAX_ITERS}" \
    eval_interval="${EVAL_INTERVAL}" eval_iters="${EVAL_ITERS}" log_interval=10 \
    compile="${COMPILE}" \
    data_dir="${DATA_DIR}" out_dir="${OUTPUT_DIR}" \
    wandb_log="${WANDB_LOG}" wandb_project="${WANDB_PROJECT}" wandb_run_name="${RUN_NAME}"
