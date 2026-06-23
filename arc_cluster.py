"""VT ARC support for submit_ngpt.py: Falcon cluster config + live free-GPU probe.

Facts gathered from the cluster (2026-06-12), see https://www.docs.arc.vt.edu/:
* This repo's home dir is shared across ARC clusters, but /scratch is
  PER-CLUSTER (Falcon and Tinkercliffs mount different filesystems). On
  Falcon, /scratch is visible from the login nodes too.
* Falcon GPU partitions / per-user QOS (sacctmgr show qos):
    l40s_normal_q  fal_l40s_normal_base  7-00:00:00  gpu<=20  (20 nodes x 4 L40S 48GB)
    a30_normal_q   fal_a30_normal_base   7-00:00:00  gpu<=64  (32 nodes x 4 A30 24GB)
    v100/t4        (Volta/Turing: no bf16 tensor cores -> unsuitable here)
* Tinkercliffs has the A100s (a100_normal_q, 14 nodes x 8, gpu<=28, 7 days)
  but is reachable only via -M/--clusters and uses its own /scratch, so the
  data would have to be prepared there separately.
* conda env: `hlm` (torch 2.11+cu128, datasets, tiktoken, simple-slurm).
"""
import getpass
import re
import subprocess

# state suffixes like "mixed-" (PLANNED: idle GPUs already promised to a
# backfill job) or "mixed*" (not responding) are NOT schedulable now
_SCHEDULABLE = {"idle", "mixed"}
_GRES_RE = re.compile(r"gpu:([\w.+-]+):(\d+)")


def _sinfo_nodes(cluster=None):
    """Yield (node, state, gres_total_str, gres_used_str), deduped by node."""
    cmd = ["sinfo", "-h", "-N",
           "-O", "NodeList:24,StateLong:16,Gres:28,GresUsed:36"]
    if cluster:
        cmd[1:1] = ["-M", cluster]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    nodes = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and "gpu:" in parts[2]:
            nodes[parts[0]] = (parts[1], parts[2], parts[3])
    return nodes


def free_gpu_snapshot(clusters=(None, "tinkercliffs")):
    """Return {(cluster, gpu_type): dict(total, free_now, nodes_with_free)}.

    free_now only counts GPUs on nodes that are schedulable right now
    (state exactly idle/mixed); nodes_with_free maps node -> free count.
    """
    snap = {}
    for cl in clusters:
        label = cl or "falcon"
        for node, (state, gres, used) in _sinfo_nodes(cl).items():
            m_tot, m_use = _GRES_RE.search(gres), _GRES_RE.search(used)
            if not m_tot:
                continue
            gtype, total = m_tot.group(1), int(m_tot.group(2))
            n_used = int(m_use.group(2)) if m_use else 0
            e = snap.setdefault((label, gtype),
                                {"total": 0, "free_now": 0, "nodes_with_free": {}})
            e["total"] += total
            if state in _SCHEDULABLE and total > n_used:
                e["free_now"] += total - n_used
                e["nodes_with_free"][node] = total - n_used
    return snap


def print_gpu_snapshot():
    snap = free_gpu_snapshot()
    print(f"{'cluster':>14} {'gpu':>8} {'total':>6} {'free now':>9}  nodes with >=2 free")
    for (cl, g), e in sorted(snap.items()):
        pairs = sorted(n for n, f in e["nodes_with_free"].items() if f >= 2)
        print(f"{cl:>14} {g:>8} {e['total']:>6} {e['free_now']:>9}  "
              f"{','.join(pairs) or '-'}")
    return snap


def make_config(repo, gpus_per_lane=4, subcluster="falcon"):
    """ARC cluster config consumed by submit_ngpt.py.

    subcluster: "falcon" (L40S) or "tinkercliffs" (A100).
    """
    user = getpass.getuser()
    scratch = f"/scratch/{user}/ngpt"
    env_act = ("source ~/.bashrc 2>/dev/null; module load Miniconda3 2>/dev/null; "
               "conda activate hlm 2>/dev/null || source activate hlm")

    if subcluster == "tinkercliffs":
        return dict(
            key="arc-tc",
            user=user,
            account="swan_research_dlm",
            scratch=scratch,
            env_act=env_act,
            max_wall_hours=168,      # tc_a100_normal_base QOS: 7-00:00:00
            lanes={i: ("a100_normal_q", "a100", gpus_per_lane, 1.0)
                   for i in range(4)},
            plan={
                0: ["gpt_fp32_10k", "gpt_fp32_50k"],
                1: ["ngpt_fp32_10k", "ngpt_fp32_50k"],
                2: ["gpt_bf16_10k", "gpt_bf16_50k"],
                3: ["ngpt_bf16_10k", "ngpt_bf16_50k"],
            },
            prep=dict(partition="a100_normal_q", gres="gpu:a100:1",
                      cpus_per_task=16, mem="64G", time="24:00:00"),
            notes=(f"Tinkercliffs /scratch is separate from Falcon.\n"
                   f"Submit from tinkercliffs login nodes (ssh shengyenc@tinkercliffs1)."),
        )

    # Default: Falcon / L40S
    return dict(
        key="arc",
        user=user,
        account="swan_research_dlm",
        scratch=scratch,
        env_act=env_act,
        max_wall_hours=168,          # fal_*_normal_base QOS: 7-00:00:00
        lanes={i: ("l40s_normal_q", "l40s", gpus_per_lane, 0.58)
               for i in range(4)},
        plan={
            0: ["gpt_fp32_10k", "gpt_fp32_50k"],
            1: ["ngpt_fp32_10k", "ngpt_fp32_50k"],
            2: ["gpt_bf16_10k", "gpt_bf16_50k"],
            3: ["ngpt_bf16_10k", "ngpt_bf16_50k"],
        },
        prep=dict(partition="t4_normal_q", gres="gpu:t4:1",
                  cpus_per_task=16, mem="64G", time="24:00:00"),
        notes=(f"/scratch IS mounted on the Falcon login nodes; if compute nodes\n"
               f"have no internet for the HF download, run the prep step there:\n"
               f"  ~/.conda/envs/hlm/bin/python data/openwebtext/prepare.py "
               f"{scratch}/data/openwebtext"),
    )
