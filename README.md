# SelectKV-MAS

**Selective KV-cache handoff for efficient latent communication in multi-agent LLM systems.**

SelectKV-MAS extends [LatentMAS](https://github.com/Gen-Verse/LatentMAS) with **SelectKV**, a training-free mechanism for reducing the KV-cache communication cost of latent multi-agent systems.

Rather than transmitting the complete newly generated KV cache at every agent handoff, SelectKV identifies and retains the most useful KV positions using complementary **relevance** and **KV-native novelty** signals. A boundary-aware design protects previously inherited context while selectively compressing newly communicable KV states.

The primary configuration, **SelectKV-90**, retains 90% of newly communicable KV positions and is evaluated against full-cache LatentMAS across six reasoning and code-generation benchmarks.

---

## Overview

Latent multi-agent systems allow agents to communicate through internal model representations rather than explicit natural-language messages. However, transferring the complete KV cache between agents can introduce substantial communication and memory overhead.

SelectKV addresses this by selectively transmitting KV states at each handoff.

The method has three main components:

- **Receiver-conditioned relevance** — estimates the importance of cached positions using cosine similarity between cached keys and the current hidden-state query/proxy.
- **KV-native novelty** — identifies distinctive information using adjacent cached-key cosine dissimilarity.
- **Boundary-aware selection** — protects KV states inherited from previous agents and applies the retention budget only to newly communicable positions.

The relevance and novelty rankings are combined using an overlap-first hybrid selection procedure.

SelectKV requires **no additional training** and operates directly on the KV cache.

---

## Repository Structure

```text
SelectKV-MAS/
├── run.py                  # Main experiment entry point and CLI
├── models.py               # Model wrapper and generation utilities
├── methods/
│   ├── baseline.py         # Single-agent baseline
│   ├── text_mas.py         # Text-based multi-agent communication
│   ├── latent_mas.py       # Full-cache LatentMAS baseline
│   └── selectkv_mas.py     # SelectKV implementation
├── selectkv/               # SelectKV utilities
├── prompts.py              # Agent prompts
├── data.py                 # Dataset loading
├── data/                   # Local benchmark data
├── requirements.txt        # Python dependencies
├── REPRODUCIBILITY.md      # Detailed reproduction instructions
└── README.md
```

The primary SelectKV implementation is located in:

```text
methods/selectkv_mas.py
```

---

## Installation

We recommend Python 3.10 and a CUDA-capable GPU with sufficient memory to run Qwen3-4B.

```bash
git clone <ANONYMOUS_REPOSITORY_URL>
cd SelectKV-MAS

conda create -n selectkv python=3.10 -y
conda activate selectkv

pip install -r requirements.txt
```

Optionally set a Hugging Face cache directory:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

Models are downloaded automatically through Hugging Face.

---

## Experimental Setup

The primary experiments use:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-4B` |
| Agent topology | Sequential |
| Latent steps | 10 |
| Batch size | 1 |
| Temperature | 0 |
| Seed | 42 |
| Samples per benchmark | 100 |
| SelectKV retention ratio | 0.90 |

We evaluate on six benchmarks:

- ARC-Easy
- ARC-Challenge
- GSM8K
- MedQA
- MBPP+
- HumanEval+

---

## Running LatentMAS

The full-cache LatentMAS baseline can be run with:

```bash
python run.py \
  --method latent_mas \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --max_samples 100 \
  --latent_steps 10 \
  --generate_bs 1 \
  --temperature 0 \
  --seed 42
```

---

## Running SelectKV

Run the primary **SelectKV-90** configuration with:

```bash
python run.py \
  --method selectkv_mas \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --prompt sequential \
  --max_samples 100 \
  --latent_steps 10 \
  --generate_bs 1 \
  --temperature 0 \
  --seed 42 \
  --selectkv_budget_ratio 0.90
```

Replace `gsm8k` with the desired task to evaluate another benchmark.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for additional experimental details.

---

## Main Results

The primary comparison evaluates full-cache **LatentMAS** against boundary-aware **SelectKV-90**.

| Benchmark | LatentMAS Acc. | SelectKV-90 Acc. | LatentMAS KV | SelectKV-90 KV | KV Reduction |
|---|---:|---:|---:|---:|---:|
| ARC-Easy | 99% | 99% | 1023.04 | 923.42 | 9.7% |
| ARC-Challenge | 90% | 90% | 1099.60 | 992.28 | 9.8% |
| GSM8K | 94% | 93% | 1068.46 | 964.29 | 9.8% |
| MedQA | 63% | 62% | 2103.52 | 1895.90 | 9.9% |
| MBPP+ | 72% | 71% | 1545.22 | 1393.25 | 9.8% |
| HumanEval+ | 89% | 83% | 1589.56 | 1433.45 | 9.8% |

Across the six benchmarks, SelectKV-90 reduces KV-cache handoff size by approximately **9.7–9.9%**, with an average accuracy change of **−1.5 percentage points** relative to full-cache LatentMAS.

---

## Ablations

The repository also supports experiments investigating the components of SelectKV.

### Boundary-aware handoff

The boundary-aware mechanism protects the inherited KV prefix from repeated pruning and applies selection only to newly communicable KV states.

On the 100-example MedQA evaluation:

| Method | Accuracy |
|---|---:|
| LatentMAS | 63% |
| SelectKV-90 without boundary protection | 58% |
| SelectKV-90 with boundary protection | 62% |

### Matched-budget selection

At the same KV communication budget on MedQA:

| Selection Method | Accuracy |
|---|---:|
| SelectKV | 62% |
| Random | 56% |
| Recent-token | 11% |

These comparisons isolate the effect of SelectKV's selection strategy from the effect of simply reducing KV-cache size.

---

## Reproducibility

Experiments use deterministic decoding (`temperature=0`) with a fixed seed (`42`).

Reported runtime can vary with GPU hardware, CUDA/PyTorch versions, and system load. Accuracy and KV-cache communication statistics provide the primary hardware-independent comparisons.

For detailed reproduction instructions, see:

**[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)**

---

## Acknowledgments

This implementation builds on the open-source **LatentMAS** framework:

> Latent Collaboration in Multi-Agent Systems  
> Original implementation: [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)

SelectKV extends the latent communication framework with selective, boundary-aware KV-cache handoff.

---

## License

This repository follows the licensing terms of the underlying LatentMAS codebase. See [`LICENSE`](LICENSE) for details.
