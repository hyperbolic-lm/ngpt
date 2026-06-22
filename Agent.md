# CLAUDE.md

## Coding Rules

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


---

## Cluster Environment 

### Unicorn Slurm

Single Cornell CoECIS SLURM cluster; `sc3379` and `ch2263` are two accounts on it.
SLURM binaries live at `/usr/local/slurm/current/bin` (prepend to `PATH`). Partition
`TIMELIMIT` is `infinite`, so a run finishes in one job; resubmitting the same
`OUTPUT_DIR` auto-resumes from `checkpoints/last.ckpt`.

**sc3379@Unicorn** — the account Claude Code sessions run as; submit SLURM directly.

- Interconnection between sc3379 and ch2263: local account here. `sbatch`/`squeue`
  work directly once `/usr/local/slurm/current/bin` is on `PATH`. To reach `ch2263`,
- Interconnection between sc3379@unicorn and shengyenc@arc
  - Use ``tailscale`` to check the IP address of the target login node and then use
  ``ssh -o "ProxyCommand=/home/sc3379/bin/tailscale --socket=/home/sc3379/.tailscale/{target_host_socket} nc %h %p" -i ~/.ssh/unicorn_internal shengyenc@{ip_address}``
  - ``{target_host_socket}`` is the ``tailscaled`` daemon socket of the host you run this from
    (its filename encodes that host — e.g. ``tailscaled-unicorn-login-02.sock``; ``ls
    ~/.tailscale`` may list several, pick the one matching ``hostname`` — only that daemon is on
    the tailnet). ``{ip_address}`` comes from ``tailscale status``; the magicDNS name also works as
    ``%h`` (``tinkercliffs1``/``tinkercliffs2``/``falcon1``/``falcon2-1`` — note falcon2's tailnet
    name is ``falcon2-1``). Use ``unicorn_internal`` (no passphrase) for non-interactive login —
    ``id_rsa`` is also authorized on ARC but passphrase-protected, so it fails under ``BatchMode``.
- identity file: not needed for local use; to SSH into `ch2263` use
  `/home/sc3379/.ssh/unicorn_internal` (public key `.pub` alongside it).
- Partitions: `thickstun,desa` (priority). Add `--exclude=desa-compute-01` — the
  2080 Ti (11 GB) OOMs at seq 1024.
- GPUs:
  - `thickstun-compute-01` — 8× RTX 6000 Ada, 48 GB   (partition `thickstun`)
  - `kuleshov-compute-02`  — 10× RTX A6000, 48 GB      (partition `desa`)
  - `kuleshov-compute-03`  — 10× RTX A5000, 24 GB      (partition `desa`)
  - `desa-compute-01`      — 8× RTX 2080 Ti, 11 GB      (partition `desa`; EXCLUDE)

**ch2263@Unicorn** — borrowed account, reached over SSH from sc3379.

- Interconnection between sc3379 and ch2263:
  `ssh -i /home/sc3379/.ssh/unicorn_internal ch2263@unicorn-login-02.coecis.cornell.edu`.
  SLURM needs a **login shell** — wrap remote commands as `ssh ... 'bash -lc "<cmd>"'`
  (`sbatch`/`squeue` are not on the non-login `PATH`).
- identity file: `/home/sc3379/.ssh/unicorn_internal` (its `.pub` is in ch2263's
  `~/.ssh/authorized_keys`).
- conda: `source ~/miniconda3/etc/profile.d/conda.sh; conda activate hodlr`.
  s-flm requires Python ≥3.10 (`dataclass(kw_only=)`); `hodlr` is py3.11 with
  torch 2.9/cu128 + lightning/hydra/transformers/datasets — leaf deps added:
  `wandb simple_slurm matplotlib scipy scikit-learn`.
- Partitions: `nlplarge`, `nlplarge-claire-highpri` (both map to one node,
  `nlplarge-compute-01`).
- GPUs:
  - `nlplarge-compute-01` — 8× A100-SXM4-80GB, 80 GB   (partitions `nlplarge`, `nlplarge-claire-highpri`)
- Storage:
  - Use ``/scratch/ch2263`` to save checkpoints and dataset. But the storage is only available on compute nodes

### ARC Slurm

Virginia Tech ARC — two relevant clusters, each with its own login nodes. ARC `/home`
and `/projects` are shared across clusters. Jobs require an allocation
(`--account=<allocation>`); GPU queues follow the `<gpu>_normal_q` / `<gpu>_preemptable_q`
naming (preemptable = lower priority, can be killed). Request GPUs with
`--partition=<queue> --gres=gpu:<n>`. (Exact submission flags: see the docs.)

Login Nodes
- tinkercliffs1, tinkercliffs2 — TinkerCliffs cluster
- falcon1, falcon2 — Falcon cluster

- Interconnection between login nodes: 
  - identity file ``~/.ssh/id_rsa``
- Interconnection between Unicorn and Arc:
  - Use ``tailscale`` to check the IP address of the target login node and then use
  ``ssh -o "ProxyCommand=/home/shengyenc/bin/tailscale --socket=/home/shengyenc/.tailscale/{target_host_socket} nc %h %p" -i ~/.ssh/id_rsa sc3379@{ip_address}``
  - ``{target_host_socket}`` is the ``tailscaled`` daemon socket of the ARC node you run this from
    (its filename encodes that node; see ``ls ~/.tailscale``). This reverse direction is not yet
    verified from this repo.

- GPUs / Partitions (source: https://www.docs.arc.vt.edu/resources/gpu.html):
  - TinkerCliffs (login: tinkercliffs1/2)
    - A100-80GB  — 112 GPUs (14 nodes × 8) | `a100_normal_q`, `a100_preemptable_q`
    - H200-141GB —  56 GPUs ( 7 nodes × 8) | `h200_normal_q`, `h200_preemptable_q`
  - Falcon (login: falcon1/2)
    - L40S-48GB — 80 GPUs  (20 nodes × 4) | `l40s_normal_q`, `l40s_preemptable_q`
    - A30-24GB  — 128 GPUs (32 nodes × 4) | `a30_normal_q`, `a30_preemptable_q`
    - V100-16GB — 80 GPUs  (40 nodes × 2) | `v100_normal_q`, `v100_preemptable_q`
    - T4-16GB   — 18 GPUs  (18 nodes × 1) | `t4_normal_q`, `t4_preemptable_q`

(ARC also exposes A100-80GB on the CUI and Biomed clusters via their own `a100_*_q`
queues — see the docs page above.)

---

## Experimental Scripts Write Up

For each experiment, create a project folder under ``experiments`` with ``{project_name}`` and store the checkpoints and results of each run under ``experiments/{project_name}/{run_name}``.

### Anaconda Environment

Follow ``README.md``: ``conda create -n sfm python=3.12`` then ``pip install -r requirements.txt``.
Python **≥3.10** is required (the code uses ``dataclass(kw_only=)``); 3.12 is the reference.

- ``torch`` / ``numpy`` are intentionally **not** pinned in ``requirements.txt`` (the NGC base
  image provides them) — install a ``torch`` build matching the node CUDA yourself
  (cu128 for the A100 / RTX-6000-Ada nodes).
- ``simple-slurm==0.3.6`` (for the sweeps) is now pinned in ``requirements.txt``, so
  ``pip install -r requirements.txt`` covers it.
- Export ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` when loading checkpoints — torch's
  ``weights_only`` default rejects the pickled Lightning ckpts.
- Reusing an existing ``sfm`` env instead of building ``sfm``, if there is no env named ``sfm``, build it via

```bash
conda create -n sfm python=3.12
conda activate sfm
pip install -r requirements.txt
```

### Training & Sampling Run Setup

- Create functions / classes in ``experiment.py`` to set up the soft link of the checkpoint for training and sampling run (if needed)
- If use ``ch2263@Unicorn``, save the checkpoints at ``/scratch/ch2263/syc_workspace/sfm_output/{project_name}/{run_name}/checkpoints`` and create a soft link at ``${OUTPUT_DIR}/checkpoints`` to link the actual checkpoint path ``/scratch/ch2263/syc_workspace/sfm_output/{project_name}/{run_name}/checkpoints`` before the training starts. This should be handle in ``experiment.py``
- The sweep.py of each experiment should use setup method in ``experiment.py`` to handle the storage of ch2263@unicorn

### Training Script

Under ``scripts/train/{dataset_name}``, 

one bash file per method ``{method}.sh`` — **one training run per script** (no sweeps or
loops inside; the sweep parameterizes it via env vars).

- Runs ``python -u -m main mode=train`` with hydra overrides, driven by env-var knobs
  (with defaults) so a sweep can set them without editing the script.
- Standard knobs: ``OUTPUT_DIR`` (= ``hydra.run.dir``; holds ``checkpoints/`` + logs),
  ``RUN_NAME``, ``WANDB_GROUP``, ``DEVICES``, ``NUM_NODES``, ``PER_GPU_BS``
  (= ``loader.batch_size``), ``GLOBAL_BATCH`` (= ``loader.global_batch_size``),
  ``MAX_STEPS``, ``CKPT_EVERY``, ``MODEL`` (model config), ``SEQ_LEN`` (= ``model.length``).
- Method knobs as needed: ``INIT``/``INIT_STD``, ``PRIOR_COV``/``RHO_MAX`` (H-FLM), ``LR``,
  ``ALPHA_MAX`` (truncation), ``SELF_COND`` (LangFlow), …
- Checkpoints → ``${OUTPUT_DIR}/checkpoints`` with ``save_top_k=1`` + ``save_last=True``.

Refer to ``scripts/train/sudoku/hflm.sh`` (and ``scripts/train/tinystories/*.sh``).

### Sampling Script

Under ``scripts/sample/{dataset_name}``

one bash file per method ``{method}.sh`` — **one eval run per script**, against a trained
checkpoint. Two passes:

- ``mode=ppl_eval`` → ``ppl.json`` (val/nll, val/ppl, val/bpd — the denoising-CE flow bound).
- ``mode=sample_eval`` → ``samples_genppl.json`` (GenPPL = gpt2-large retokenized generative
  perplexity, sample entropy, generated text).
- Loads the checkpoint via ``eval.checkpoint_path=${CKPT_PATH}`` (``eval.strict_loading=false``).
- Knobs: ``CKPT_PATH``, ``OUTPUT_DIR``, ``MODEL``, ``SEQ_LEN``, ``EVAL_BS``, sampler
  ``STEPS``/``VELOCITY``/``TOPK_VELOCITY``. ``MODEL`` and ``SEQ_LEN`` MUST match the training
  run so the architecture matches the checkpoint.
- Always single-GPU: ``DEVICES=1`` + ``CUDA_VISIBLE_DEVICES=0`` (see SLURM Env Setup).

Refer to ``scripts/sample/sudoku/hflm.sh`` (and ``scripts/sample/tinystories/*.sh``).

### Sweep Script

Under ``experiments/{project_name}``

One experiment project script for one project

Use python and ``simple_slurm`` to submit jobs to slurm

**Orchestration only** — the sweep CALLS the train + sample scripts; it never inlines
``python -m main``. It builds the parameter grid, submits one SLURM job per cell (train
then eval), and is idempotent/resumable: skip a cell whose ``eval/ppl.json`` exists or
whose job name is already in ``squeue``.

Unique and semanticful name for each slurm job, ex ``{project_name}_{var1}-{var1_value}_{var2}-{var2_value}...``, use abbr for each variable nam ``var1``, followed by a parameter value ``var1_value``. Only record the searched parameters in the project

### SLURM Env Setup

The sweep submits one SLURM job per run with `simple_slurm`; each job activates the
conda env (see `## Cluster Environment`), then runs the train script followed by the
sample/eval script. Boilerplate every job body needs:

```bash
# simple_slurm resources: partition=<cluster queue>, gres=gpu:<N>, ntasks=1,
#   cpus_per_task=16, mem=128G, time=..., output=experiments/<project>/logs/<run>_%j.log
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export SLURM_JOB_NAME=bash                      # Lightning uses its own DDP launcher, not srun
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1     # avoid NCCL hangs on single-/multi-GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=<conda-env>/bin:$PATH               # sc3379: .../envs/sfm ; ch2263: .../envs/hodlr
cd <REPO>
OUTPUT_DIR=... DEVICES=<N> PER_GPU_BS=<b> MAX_STEPS=... [MODEL=... SEQ_LEN=... GLOBAL_BATCH=...] \
    bash scripts/train/<dataset>/<method>.sh
CKPT_PATH=<out>/checkpoints/last.ckpt OUTPUT_DIR=<out>/eval DEVICES=1 \
    bash scripts/sample/<dataset>/<method>.sh
```

- **Gradient accumulation** is derived by the config, not set by hand:
  `accumulate_grad_batches = global_batch_size / (devices × per_gpu_batch × num_nodes)`.
  Pick `GLOBAL_BATCH`/`DEVICES`/`PER_GPU_BS` so it divides evenly — the dataloader
  asserts `global_batch == per_gpu_batch × num_nodes × num_gpus × accum`.
- **Eval on one GPU** (`DEVICES=1`; the sample scripts set `CUDA_VISIBLE_DEVICES=0`) so
  `torch.cuda.device_count()` matches `trainer.devices` and the assert holds.
- **Idempotent + resumable**: skip a cell if its `eval/ppl.json` exists or its job name
  is already in `squeue`; resubmitting the same `OUTPUT_DIR` auto-resumes from `last.ckpt`.

### Training and Sampling Outputs

All checkpoints and generated text go under ``outputs/{project_name}`` (same name as the
experiment folder under ``experiments/``), one subfolder per run:

```
outputs/{project_name}/{run_name}/          # = hydra.run.dir of the training run
├── checkpoints/
│   ├── last.ckpt                            # resume + eval load this (save_last=True)
│   └── ...                                  # periodic step checkpoint(s), capped by save_top_k=1
├── .hydra/ , *.log , wandb/                 # training config + logs
└── eval/
    ├── ppl.json                             # numerical results: val/nll, val/ppl, val/bpd
    ├── samples_genppl.json                  # GenPPL (gen_ppl_first_chunk_retok), entropy, avg_nfe, text
    ├── ppl/                                 # hydra run dir for the ppl_eval pass (config/logs)
    └── sample/                              # hydra run dir for the sample_eval pass (config/logs)
```

- ``{run_name}`` = the semantic SLURM job name (e.g. ``{var}-{val}_...``).
- ``ppl.json`` holds the ``trainer.validate()`` metrics (the denoising-CE flow bound — not a
  true AR PPL). ``samples_genppl.json`` is the generation-quality deliverable; read GenPPL
  **with** entropy (low GenPPL + low entropy ⇒ degenerate/repetitive collapse, not quality).
- ``experiments/report.py {project_name}`` scans every ``{run_name}/eval/`` and writes a
  summary table to ``experiments/{project_name}/RESULTS.md``.

### Reports and Experiment

- For each experimental project, create a experimental arch file called ``experiments/{project_name}/EXPERIMENT.md`` to record the experimental design, hypothesis, GPU allocation, and expected wall clock time
- For each experimental project, create a experimental report, called ``experiments/{project_name}/RESULTS.md`` to record the numerical and qualitative results of each experiment. Provide table for the numerical results and inisights and conclusions of the experiment.