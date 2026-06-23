#!/usr/bin/env python3
"""Sequential GPT/nGPT 0.5B sweep via simple-slurm, on one of two clusters.

Usage:
    python submit_ngpt.py --cluster unicorn            # Cornell Unicorn (A100)
    python submit_ngpt.py --cluster arc                # VT ARC Falcon (L40S)
    python submit_ngpt.py --check-gpus                 # ARC: live free-GPU table, no submit
    python submit_ngpt.py --dry-run                    # print job scripts, submit nothing
The cluster is auto-detected from the hostname when --cluster is omitted.
On ARC use an env with simple-slurm, e.g.  ~/.conda/envs/hlm/bin/python submit_ngpt.py

Design notes
------------
* All heavy artifacts (HF cache, openwebtext bins, checkpoints, stat files)
  go to the cluster's /scratch. On Unicorn /scratch/ch2263 is only mounted on
  the nlplarge / nlplarge-claire-highpri compute nodes, so a one-time
  `prep_data` job (running ON the partition) creates the directory tree and
  tokenizes OpenWebText if train.bin/val.bin are missing; every training lane
  depends on it (afterok). On ARC Falcon /scratch/<user> is per-cluster but
  also visible from the login nodes. slurm .out/.err logs stay in the repo
  under slurm_logs/ (home dir) on both clusters.
* GPUs are modeled as independent "lanes"; jobs within a lane run in
  sequence via --dependency, lanes run concurrently. If fewer GPUs are free,
  lanes simply wait in queue; nothing breaks.
    unicorn: 3 lanes x 2 A100 (2 lanes highpri, 1 on preemptible nlplarge)
    arc:     4 lanes x 4 L40S on l40s_normal_q (16 <= 20 GPU/user QOS cap;
             use --gpus-per-lane 2 for a lighter 8-GPU footprint)
* Schedules are ordered so results land as early as possible: every lane
  starts with a 10k run; val loss is printed/checkpointed every 1000 iters,
  so curves are visible in the logs and in <run>/stat long before a job ends.
* Long runs are split into <=max_wall_hours chunks chained with afterany:
  each chunk resumes from ckpt.pt, stops time_limit_seconds before the wall
  limit, and exits immediately if the run already wrote its `finished`
  marker. requeue=True additionally survives preemption.
* train.py/model.py were patched to expose --weight_dtype (bfloat16|float32
  storage of matrix weights; embeddings stay fp32 as in the README setup),
  --out_dir and --data_dir. Autocast compute stays bfloat16 for both
  precisions, matching the paper's reimplementation notes.
* gradient_accumulation_steps is the GLOBAL value (train.py divides by world
  size): 8 micro-batch x 64 accum = 512 seqs x 1024 tok = 0.5M tokens/iter,
  the paper's setup. 64 is divisible by 2 or 4 GPUs per lane.
* est. hours are calibrated on 2xA100; other lanes scale them by
  2 / (n_gpus * speed), with speed = per-GPU throughput relative to A100.
"""
import argparse
import math
import os
import socket
import sys

REPO     = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = f"{REPO}/slurm_logs"
MICRO_BATCH  = 8
GLOBAL_ACCUM = 64          # global batch = MICRO_BATCH * GLOBAL_ACCUM = 512 sequences
LR           = 3e-3

# name -> (use_nGPT, weight_dtype, max_iters, est hours on a 2xA100 lane)
RUN_SPECS = {
    "gpt_fp32_10k":  (0, "float32",  10_000,  36),
    "gpt_fp32_50k":  (0, "float32",  50_000, 180),
    "gpt_bf16_10k":  (0, "bfloat16", 10_000,  30),
    "gpt_bf16_50k":  (0, "bfloat16", 50_000, 150),
    "ngpt_fp32_10k": (1, "float32",  10_000,  65),
    "ngpt_fp32_50k": (1, "float32",  50_000, 330),
    "ngpt_bf16_10k": (1, "bfloat16", 10_000,  55),
    "ngpt_bf16_50k": (1, "bfloat16", 50_000, 270),
}


def unicorn_config():
    """Cornell Unicorn: 3-6 A100 on nlplarge / nlplarge-claire-highpri."""
    user = "ch2263"
    # exact GRES type on nlplarge-compute-01 ("a100" is only a feature)
    gtype = "nvidia_a100-sxm4-80gb"
    return dict(
        key="unicorn",
        user=user,
        scratch=f"/scratch/{user}/ngpt",
        env_act="source ~/.bashrc && conda activate hfm",
        max_wall_hours=48,
        lanes={
            0: ("nlplarge-claire-highpri", gtype, 2, 1.0),
            1: ("nlplarge-claire-highpri", gtype, 2, 1.0),
            2: ("nlplarge",                gtype, 2, 1.0),
        },
        # fp32 before bf16 (user priority), 10k before 50k within a lane
        plan={
            0: ["gpt_fp32_10k", "gpt_fp32_50k", "gpt_bf16_50k"],
            1: ["ngpt_fp32_10k", "gpt_bf16_10k", "ngpt_bf16_10k", "ngpt_bf16_50k"],
            2: ["ngpt_fp32_50k"],   # preemptible lane: longest run starts at once
        },
        prep=dict(partition="nlplarge-claire-highpri", cpus_per_task=16,
                  mem="64G", time="24:00:00"),
        notes="/scratch is NOT visible from the Unicorn login node; monitor via logs.",
    )


def detect_cluster():
    host = socket.gethostname()
    if host.startswith(("tinkercliffs", "tc-")):
        return "arc-tc"
    if host.startswith(("falcon", "fal")):
        return "arc"
    return "unicorn"


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--cluster", choices=["unicorn", "arc", "arc-tc"],
                    default=detect_cluster(),
                    help="target cluster (default: auto-detect from hostname). "
                         "arc=Falcon/L40S, arc-tc=Tinkercliffs/A100")
parser.add_argument("--gpus-per-lane", type=int, default=4, choices=[2, 4],
                    help="ARC only: GPUs per lane (default 4)")
parser.add_argument("--check-gpus", action="store_true",
                    help="ARC only: print the live free-GPU table and exit")
parser.add_argument("--dry-run", action="store_true",
                    help="print job scripts without submitting")
parser.add_argument("--only", nargs="+", metavar="RUN",
                    help="submit only the named runs (e.g. --only ngpt_fp32_10k gpt_bf16_10k)")
args = parser.parse_args()

if args.cluster in ("arc", "arc-tc"):
    import arc_cluster
    if args.check_gpus:
        arc_cluster.print_gpu_snapshot()
        sys.exit(0)
    subcluster = "tinkercliffs" if args.cluster == "arc-tc" else "falcon"
    CFG = arc_cluster.make_config(REPO, gpus_per_lane=args.gpus_per_lane,
                                  subcluster=subcluster)
    print("Live GPU availability:")
    arc_cluster.print_gpu_snapshot()
    print()
else:
    if args.check_gpus:
        sys.exit("--check-gpus is only implemented for ARC clusters")
    CFG = unicorn_config()

from simple_slurm import Slurm  # noqa: E402  (after --check-gpus fast path)

DATA_DIR = f"{CFG['scratch']}/data/openwebtext"   # train.bin/val.bin
RUN_ROOT = f"{CFG['scratch']}/runs"               # ckpt.pt/stat/args/finished per run
HF_CACHE = f"{CFG['scratch']}/hf_cache"           # raw HF download cache (~80GB)
ENV_ACT  = CFG["env_act"]
MAX_WALL_HOURS = CFG["max_wall_hours"]

os.makedirs(LOG_DIR, exist_ok=True)


def submit(slurm, script, what):
    if args.dry_run:
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
ACCOUNT = {"account": CFG["account"]} if CFG.get("account") else {}
prep = Slurm(
    job_name="owt_prep",
    nodes=1, ntasks_per_node=1,
    output=f"{LOG_DIR}/owt_prep_%j.out", error=f"{LOG_DIR}/owt_prep_%j.err",
    **CFG["prep"], **ACCOUNT,
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
for lane, names in CFG["plan"].items():
    part, gtype, ngpus, speed = CFG["lanes"][lane]
    assert GLOBAL_ACCUM % ngpus == 0
    for name in names:
        if args.only and name not in args.only:
            continue
        use_ngpt, wdtype, iters, base_hours = RUN_SPECS[name]
        est_hours = base_hours * 2.0 / (ngpus * speed)
        n_chunks = max(1, math.ceil(est_hours * 1.25 / MAX_WALL_HOURS))  # 25% margin
        script = train_command(name, use_ngpt, wdtype, iters, ngpus)

        job_ids = []
        for chunk in range(n_chunks):
            kw = dict(
                job_name=f"{name}_c{chunk}",
                partition=part, nodes=1, ntasks_per_node=1,
                cpus_per_task=8 * ngpus, mem=f"{60 * ngpus}G",
                gres=f"gpu:{gtype}:{ngpus}",
                time=f"{MAX_WALL_HOURS // 24}-{MAX_WALL_HOURS % 24:02d}:00:00",
                output=f"{LOG_DIR}/{name}_c{chunk}_%j.out",
                error=f"{LOG_DIR}/{name}_c{chunk}_%j.err",
                requeue=True,  # survive preemption
                **ACCOUNT,
            )
            if lane in last_in_lane:
                # afterany: keep the chain moving even if a chunk dies/times
                # out; the next chunk just resumes from the last checkpoint.
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
{CFG['notes']}

Monitoring (from the login node):
  squeue -u {CFG['user']}                            # queue state
  tail -f {LOG_DIR}/<run>_c0_*.out                   # live loss curve
Val loss is evaluated every 1000 iters; the per-run history is in
{RUN_ROOT}/<run>/stat (col 1 iter, col 3/4 train/val loss).
ETAs above EXCLUDE the one-time data prep job and any queue wait time.
""")
