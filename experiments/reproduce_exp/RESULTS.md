# nGPT Reproduction Results

Reproduction of the nGPT paper (Loshchilov et al., ICLR 2025) using the authors'
nanoGPT-based codebase. All runs use 0.5B parameter models (24 layers, 16 heads,
d=1024) trained on OpenWebText with GPT-2 BPE tokenizer (50k vocab), context
length 1024, global batch size 0.5M tokens/iter, lr=3e-3 with cosine decay.

## Setup

| Parameter | Value |
|---|---|
| Model size | 0.5B (n_layer=24, n_head=16, n_embd=1024) |
| Dataset | OpenWebText (GPT-2 BPE, 50k vocab) |
| Context length | 1024 |
| Batch size | 512 seqs × 1024 tok = 0.5M tokens/iter |
| Learning rate | 3e-3, cosine decay to 0 |
| GPT weight decay | 0.1 |
| nGPT weight decay | 0.0 |
| GPT warmup | 2000 iters |
| nGPT warmup | 0 iters |
| Attention | SDPA (flash_attn not installed) |
| Compile | False |

Clusters: VT ARC Falcon (L40S-48GB, 4 GPUs/run) and TinkerCliffs (A100-80GB,
4 GPUs/run). Runs were distributed across clusters based on GPU availability.

## Final Val Loss

| Run | Model | Weight Dtype | Iters | Cluster | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| gpt_fp32_10k | GPT | float32 | 10k | Falcon (L40S) | 2.8227 | 2.8352 |
| gpt_bf16_10k | GPT | bfloat16 | 10k | TinkerCliffs (A100) | 2.8564 | 2.8680 |
| ngpt_fp32_10k | nGPT | float32 | 10k | TinkerCliffs (A100) | 2.7325 | 2.7453 |
| ngpt_bf16_10k | nGPT | bfloat16 | 10k | TinkerCliffs (A100) | 2.7474 | 2.7592 |
| gpt_fp32_50k | GPT | float32 | 50k | TinkerCliffs (A100) | 2.6261 | 2.6617 |
| gpt_bf16_50k | GPT | bfloat16 | 50k | Falcon (L40S) | 2.7120 | 2.7446 |
| ngpt_fp32_50k | nGPT | float32 | 50k | Falcon (L40S) | 2.5781 | 2.6124 |
| ngpt_bf16_50k | nGPT | bfloat16 | 50k | Falcon (L40S) | 2.6083 | 2.6387 |

## Learning Curves (Val Loss at Eval Checkpoints)

### GPT fp32 (10k: Falcon, 50k: TinkerCliffs)

| Iter | 1k | 2k | 5k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|---|---|
| 10k run | 3.709 | 3.397 | 3.082 | **2.835** | — | — | — | — |
| 50k run | 3.704 | 3.399 | 3.128 | 3.009 | 2.924 | 2.820 | 2.715 | **2.662** |

### GPT bf16 (10k: TinkerCliffs, 50k: Falcon)

| Iter | 1k | 2k | 5k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|---|---|
| 10k run | 3.708 | 3.382 | 3.053 | **2.868** | — | — | — | — |
| 50k run | 3.712 | 3.381 | 3.084 | 2.942 | 2.848 | 2.777 | 2.750 | **2.745** |

### nGPT fp32 (10k: TinkerCliffs, 50k: Falcon)

| Iter | 1k | 2k | 5k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|---|---|
| 10k run | 3.594 | 3.401 | 3.138 | **2.745** | — | — | — | — |
| 50k run | 3.595 | 3.429 | 3.375 | 3.305 | 3.150 | 2.924 | 2.702 | **2.612** |

### nGPT bf16 (10k: TinkerCliffs, 50k: Falcon)

| Iter | 1k | 2k | 5k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|---|---|
| 10k run | 3.588 | 3.391 | 3.135 | **2.759** | — | — | — | — |
| 50k run | 3.602 | 3.427 | 3.381 | 3.300 | 3.160 | 2.924 | 2.705 | **2.639** |

## Speedup Analysis

The paper claims ~4× training token reduction for nGPT over GPT at context length
1024 (0.5B model, Figure 4). Speedup = (GPT iters to reach loss L) / (nGPT iters
to reach loss L).

**At val loss ≈ 2.745 (nGPT fp32 10k final):**
- nGPT fp32 reaches 2.745 at 10k iters
- GPT fp32 reaches 2.745 between 30k–40k iters (interpolated: ~37.1k)
- **Speedup ≈ 3.7×**

**At val loss ≈ 2.662 (GPT fp32 50k final):**
- GPT fp32 reaches 2.662 at 50k iters
- nGPT fp32 reaches 2.662 between 40k–50k iters (interpolated: ~44.5k)
- **Speedup ≈ 1.1×**

The speedup is loss-level–dependent: large at moderate loss (early-to-mid training),
diminishing as both models converge toward the data entropy limit. At the 2.745 loss
level, our 3.7× is consistent with the paper's ~4× claim.

## Precision (bf16 vs fp32)

| Model | Iters | fp32 Val | bf16 Val | Gap |
|---|---|---|---|---|
| GPT | 10k | 2.835 | 2.868 | +0.033 |
| GPT | 50k | 2.662 | 2.745 | +0.083 |
| nGPT | 10k | 2.745 | 2.759 | +0.014 |
| nGPT | 50k | 2.612 | 2.639 | +0.027 |

nGPT is more robust to bf16 weight quantization than GPT. The bf16 → fp32 gap for
GPT grows from 0.033 at 10k to 0.083 at 50k (precision errors accumulate over
training), while nGPT's gap stays small (0.014 → 0.027). The hypersphere
normalization in nGPT constrains weight magnitudes, limiting the damage from reduced
precision.

## Comparison to Paper

| Aspect | Paper | Reproduction | Note |
|---|---|---|---|
| Speedup (1k ctx) | ~4× | ~3.7× at moderate loss | Consistent at comparable loss levels |
| nGPT < GPT val loss | Yes | Yes, at all settings | nGPT consistently outperforms GPT |
| Tokenizer | LLaMA-2 (32k vocab) | GPT-2 BPE (50k vocab) | Larger vocab → more parameters in embeddings |
| Attention | flash_attn | SDPA fallback | Mathematically equivalent, slower wall-clock |
| torch.compile | True | False | No effect on loss, only on speed |

**Key takeaways:**
1. nGPT consistently outperforms GPT across all dtype and iteration settings.
2. The ~4× speedup claim is reproducible at moderate loss levels (val ≈ 2.7–2.8);
   it diminishes at lower loss levels where both models approach the data limit.
3. nGPT is substantially more robust to bf16 weight precision than GPT, likely
   due to hypersphere normalization constraining weight magnitudes.
4. The best overall result is nGPT fp32 50k at val loss 2.612, beating GPT fp32
   50k (2.662) by 0.050 — a meaningful gap at this scale.
