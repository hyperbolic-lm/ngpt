#!/usr/bin/env python3
"""Sequential GPT/nGPT 0.5B sweep on the Cornell Unicorn cluster via simple-slurm.

Usage (on the unicorn login node, in any env with simple-slurm):
    pip install simple-slurm
    python submit_ngpt.py            # submit everything
    python submit_ngpt.py --dry-run  # print job scripts without submitting

Design notes
------------
* All heavy artifacts (HF cache, openwebtext bins, checkpoints, stat files) go
  to /scratch/ch2263, which is only mounted on the nlplarge /
  nlplarge-claire-highpri compute nodes. Therefore:
    - a one-time `prep_data` job (running ON the partition) creates the
      /scratch directory tree and tokenizes OpenWebText if train.bin/val.bin
      are missing; every training lane depends on it (afterok).
    - slurm .out/.err logs stay in the repo under slurm_logs/ (home dir):
      slurmd must be able to open the log file at job start and sbatch runs
      on the login node where /scratch is not visible. Logs are only a few
      MB total.
* 3-6 A100s are modeled as 3 lanes x 2 GPUs (2 on each highpri, 1 lane on
  nlplarge). If fewer GPUs are free, lanes simply wait in queue; nothing
  breaks. Jobs within a lane run in sequence via --dependency.
* fp32 experiments are scheduled before bf16 ones (user priority), and 10k
  runs before 50k runs so comparisons land as early as possible. Val loss is
  printed/checkpointed every 1000 iters, so curves are visible in the logs
  and in <run>/stat long before a job finishes.
* Long runs are split into <=MAX_WALL_HOURS chunks chained with afterany:
  each chunk resumes from ckpt.pt, stops time_limit_seconds before the wall
  limit, and exits immediately if the run already wrote its `finished`
  marker. requeue=True additionally survives preemption on nlplarge.
* train.py/model.py were patched to expose --weight_dtype (bfloat16|float32
  storage of matrix weights; embeddings stay fp32 as in the README setup),
  --out_dir and --data_dir. Autocast compute stays bfloat16 for both
  precisions, matching the paper's reimplementation notes.
* gradient_accumulation_steps is the GLOBAL value (train.py divides by world
  size): 8 micro-batch x 64 accum = 512 seqs x 1024 tok = 0.5M tokens/iter,
  the paper's setup. 64 is divisible by 2 or 4 GPUs per lane.
"""
import math
import os
import sys

from simple_slurm import Slurm

DRY_RUN = "--dry-run" in sys.argv

# ---------------- user config ----------------
NETID      = "ch2263"
REPO       = os.path.dirname(os.path.abspath(__file__))
SCRATCH    = f"/scratch/{NETID}/ngpt"
DATA_DIR   = f"{SCRATCH}/data/openwebtext"     # train.bin/val.bin
RUN_ROOT   = f"{SCRATCH}/runs"                 # ckpt.pt/stat/args/finished per run
HF_CACHE   = f"{SCRATCH}/hf_cache"             # raw HF download cache (~80GB)
ENV_ACT    = "source ~/.bashrc && conda activate hfm"
LOG_DIR    = f"{REPO}/slurm_logs"
GPU_TYPE   = "nvidia_a100-sxm4-80gb"  # exact GRES type on nlplarge-compute-01 ("a100" is only a feature)
MAX_WALL_HOURS = 48          # per-chunk walltime; lower it if the partition limit is smaller
MICRO_BATCH    = 8
GLOBAL_ACCUM   = 64          # global batch = MICRO_BATCH * GLOBAL_ACCUM = 512 sequences
LR             = 3e-3

os.makedirs(LOG_DIR, exist_ok=True)

# lane -> (partition, n_gpus). GLOBAL_ACCUM must be divisible by n_gpus.
LANES = {
    0: ("nlplarge-claire-highpri", 2),
    1: ("nlplarge-claire-highpri", 2),
    2: ("nlplarge",                2),
}

# (name, use_nGPT, weight_dtype, max_iters, lane, est_hours)
# List order = execution order within each lane. fp32 first, then bf16;
# within each precision the 10k runs come before the 50k runs.
RUNS = [
    # lane 0 (highpri): fp32 GPT chain, then leftover bf16 50k
    ("gpt_fp32_10k",  0, "float32",  10_000, 0,  36),
    ("gpt_fp32_50k",  0, "float32",  50_000, 0, 180),
    ("gpt_bf16_50k",  0, "bfloat16", 50_000, 0, 150),
    # lane 1 (highpri): fp32 nGPT 10k first, then the bf16 runs
    ("ngpt_fp32_10k", 1, "float32",  10_000, 1,  65),
    ("gpt_bf16_10k",  0, "bfloat16", 10_000, 1,  30),
    ("ngpt_bf16_10k", 1, "bfloat16", 10_000, 1,  55),
    ("ngpt_bf16_50k", 1, "bfloat16", 50_000, 1, 270),
    # lane 2 (nlplarge, preemptible): longest fp32 run starts immediately
    ("ngpt_fp32_50k", 1, "float32",  50_000, 2, 330),
]


def submit(slurm, script, what):
    if DRY_RUN:
        print(f"\n===== {what} =====\n{slurm}\n{script}")
        return 0
    return slurm.sbatch(script)


# ---------------- one-time data prep ----------------
prep_script = f"""{ENV_ACT}
set -x
mkdir -p {DATA_DIR} {RUN_ROOT} {HF_CACHE}
if [ -f {DATA_DIR}/train.bin ] && [ -f {DATA_DIR}/val.bin ]; then
    echo "openwebtext already prepared, nothing to do"; exit 0
fi
export HF_HOME={HF_CACHE}
export HF_DATASETS_CACHE={HF_CACHE}/datasets
cd {REPO}
python data/openwebtext/prepare.py {DATA_DIR}
"""
prep = Slurm(
    job_name="owt_prep",
    partition="nlplarge-claire-highpri",
    nodes=1, ntasks_per_node=1, cpus_per_task=16, mem="64G",
    time="24:00:00",
    output=f"{LOG_DIR}/owt_prep_%j.out", error=f"{LOG_DIR}/owt_prep_%j.err",
    # if the partition rejects CPU-only jobs, uncomment:
    # gres=f"gpu:{GPU_TYPE}:1",
)
prep_id = submit(prep, prep_script, "owt_prep")
print(f"data prep -> job {prep_id}")


# ---------------- training jobs ----------------
def train_command(name, use_ngpt, wdtype, iters, ngpus):
    run_dir = f"{RUN_ROOT}/{name}"
    wd, warmup = (0.0, 0) if use_ngpt else (0.1, 2000)
    # leave 30 min of the wall limit for the final eval + checkpoint write
    time_limit = MAX_WALL_HOURS * 3600 - 1800
    return f"""{ENV_ACT}
set -x
export OMP_NUM_THREADS=8
cd {REPO}
RUN_DIR={run_dir}
mkdir -p $RUN_DIR
if [ -f $RUN_DIR/finished ]; then echo "{name} already finished"; exit 0; fi
if [ -f $RUN_DIR/ckpt.pt ]; then INIT=resume; else INIT=scratch; fi
torchrun --nnodes=1 --nproc_per_node={ngpus} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train.py \\
  --init_from=$INIT --use_nGPT={use_ngpt} --weight_dtype={wdtype} --dtype=bfloat16 \\
  --learning_rate={LR} --weight_decay={wd} --warmup_iters={warmup} \\
  --n_layer=24 --n_head=16 --n_embd=1024 --block_size=1024 \\
  --batch_size={MICRO_BATCH} --gradient_accumulation_steps={GLOBAL_ACCUM} \\
  --max_iters={iters} --lr_decay_iters={iters} --min_lr=0.0 \\
  --eval_interval=1000 --eval_iters=200 --log_interval=10 --compile=False \\
  --time_limit_seconds={time_limit} \\
  --out_dir=$RUN_DIR --data_dir={DATA_DIR}
"""


last_in_lane = {}     # lane -> last submitted job id
lane_eta = {}         # lane -> cumulative estimated hours
print(f"\n{'run':>15} {'lane':>4} {'partition':>24} {'chunks':>6} {'est ETA':>10}   job ids")
for name, use_ngpt, wdtype, iters, lane, est_hours in RUNS:
    part, ngpus = LANES[lane]
    assert GLOBAL_ACCUM % ngpus == 0
    n_chunks = max(1, math.ceil(est_hours * 1.25 / MAX_WALL_HOURS))  # 25% margin
    script = train_command(name, use_ngpt, wdtype, iters, ngpus)

    job_ids = []
    for chunk in range(n_chunks):
        kw = dict(
            job_name=f"{name}_c{chunk}",
            partition=part, nodes=1, ntasks_per_node=1,
            cpus_per_task=8 * ngpus, mem=f"{60 * ngpus}G",
            gres=f"gpu:{GPU_TYPE}:{ngpus}",
            time=f"{MAX_WALL_HOURS // 24}-{MAX_WALL_HOURS % 24:02d}:00:00",
            output=f"{LOG_DIR}/{name}_c{chunk}_%j.out",
            error=f"{LOG_DIR}/{name}_c{chunk}_%j.err",
            requeue=True,  # survive preemption (esp. on nlplarge)
        )
        if lane in last_in_lane:
            # afterany: keep the chain moving even if a chunk dies/times out;
            # the next chunk just resumes from the last checkpoint.
            kw["dependency"] = dict(afterany=last_in_lane[lane])
        elif prep_id:
            kw["dependency"] = dict(afterok=prep_id)
        job_id = submit(Slurm(**kw), script, f"{name} chunk {chunk}")
        job_ids.append(job_id)
        last_in_lane[lane] = job_id

    lane_eta[lane] = lane_eta.get(lane, 0) + est_hours
    eta = lane_eta[lane]
    print(f"{name:>15} {lane:>4} {part:>24} {n_chunks:>6} {eta:>7.0f}h ({eta/24:.1f}d)  {job_ids}")

print(f"""
Monitoring (from the login node):
  squeue -u {NETID}                                  # queue state
  tail -f {LOG_DIR}/<run>_c0_*.out                   # live loss curve
Val loss is evaluated every 1000 iters; from any job on the partition the
per-run history is in {RUN_ROOT}/<run>/stat (col 1 iter, col 3/4 train/val loss).
ETAs above EXCLUDE the one-time data prep job and any queue wait time.
""")
