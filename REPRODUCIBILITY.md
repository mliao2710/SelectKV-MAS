# Reproducibility

This repository contains the implementation used for the experiments in **Selective and Hybrid-State Handoff for Latent Multi-Agent Systems**. The code builds on LatentMAS and implements **SelectKV**, a selective KV-cache handoff mechanism for reducing latent multi-agent communication cost while preserving task performance.

## Environment

The experiments use **Qwen/Qwen3-4B** with the Hugging Face backend.

Create the environment and install the dependencies with:

```bash
conda create -n selectkv python=3.10 -y
conda activate selectkv
pip install -r requirements.txt
```

The model is downloaded automatically from Hugging Face. A GPU with sufficient memory to run Qwen3-4B is required.

## Main Experimental Configuration

Unless otherwise specified, the reported experiments use:

- Model: `Qwen/Qwen3-4B`
- Prompt topology: sequential
- Latent steps: 10
- Batch size: 1
- Temperature: 0
- Seed: 42
- SelectKV retention ratio: 0.90
- Number of evaluated examples per benchmark: 100

The main benchmarks are:

- ARC-Easy
- ARC-Challenge
- GSM8K
- MedQA
- MBPP+
- HumanEval+

## SelectKV

SelectKV reduces communication overhead by selectively retaining KV-cache positions during latent handoff between agents.

The method uses two complementary signals:

1. **Receiver-conditioned relevance**, computed using cosine similarity between cached keys and the current hidden-state query/proxy.
2. **KV-native novelty**, computed from adjacent cached-key cosine dissimilarity.

The two rankings are combined using an overlap-first hybrid selection procedure.

SelectKV is also **boundary-aware**. KV positions inherited from the previous agent are treated as a protected prefix. The retention budget is applied only to newly communicable KV positions, preventing repeated pruning of information that has already survived previous handoffs.

For the primary configuration, SelectKV retains 90% of the newly communicable KV positions.

The main implementation is located in:

```text
methods/selectkv_mas.py
```

The command-line integration and experimental configuration are defined in:

```text
run.py
```

## Running the LatentMAS Baseline

A baseline LatentMAS experiment can be run with:

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

## Running SelectKV

The corresponding SelectKV experiment at the primary 90% retention operating point can be run with:

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

Replace `gsm8k` with the desired benchmark task to reproduce results on the other datasets.

## Evaluation

For each benchmark, we report task accuracy together with KV-cache communication statistics and wall-clock runtime.

The primary comparison is between:

- **LatentMAS:** full KV-cache handoff
- **SelectKV-90:** boundary-aware SelectKV with a 0.90 retention ratio

The paper additionally evaluates boundary handling and matched-budget selection strategies to isolate the contribution of SelectKV's selection mechanism.

## Reproducibility Notes

All primary experiments use deterministic decoding (`temperature=0`) and a fixed random seed (`seed=42`).

Runtime can vary depending on GPU hardware, CUDA/PyTorch versions, and system load. Accuracy and KV-cache retention statistics are the primary hardware-independent quantities for comparison.

This repository contains the implementation necessary to run both the LatentMAS baseline and SelectKV under the same experimental framework.
