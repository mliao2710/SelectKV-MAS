import argparse
import json
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.selectkv_mas import SelectKVMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time
import torch


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas", "selectkv_mas"], required=True,
                        help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"],
                        help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-14B').")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
    parser.add_argument(
        "--force_eager_attention",
        action="store_true",
        help="Force eager attention for fair attention-backend controls."
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # SelectKV
    parser.add_argument(
        "--selectkv_budget_ratio",
        type=float,
        default=0.80,
        help="Fraction of candidate KV positions retained by SelectKV."
    )
    parser.add_argument(
        "--selectkv_recent_tokens",
        type=int,
        default=4,
        help="Number of most recent KV positions protected from pruning."
    )
    parser.add_argument(
        "--selectkv_overlap_pool_fraction",
        type=float,
        default=0.50,
        help="Top-ranked fraction used when finding relevance/novelty overlap."
    )
    parser.add_argument(
        "--selectkv_adaptive",
        action="store_true",
        help="Adapt SelectKV retention per handoff using score entropy and ranking agreement."
    )
    parser.add_argument(
        "--selectkv_adaptive_aggressive",
        action="store_true",
        help="Use a more aggressive boundary-aware adaptive SelectKV policy."
    )
    parser.add_argument(
        "--selectkv_adaptive_min_ratio",
        type=float,
        default=0.85,
        help="Minimum retention ratio used by adaptive SelectKV."
    )

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()
    
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )

    elif args.method == "selectkv_mas":
        method = SelectKVMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
        )
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)

    # ---------------------------------------------------------
    # Aggregate efficiency metrics across evaluated problems.
    # Paper experiments use generate_bs=1 so cache metrics are
    # attributed exactly to individual problems.
    # ---------------------------------------------------------
    n_samples = len(preds)

    output_tokens_total = sum(
        int(p.get("output_tokens", 0))
        for p in preds
    )

    # TextMAS-only communication metric. Other methods naturally
    # return zero because they do not exchange natural-language
    # messages between agents.
    text_comm_tokens_total = sum(
        int(p.get("text_comm_tokens", 0))
        for p in preds
    )

    kv_positions_handoff_total = sum(
        int(p.get("kv_positions_handoff", 0))
        for p in preds
    )

    logical_kv_payload_bytes_total = sum(
        int(p.get("logical_kv_payload_bytes", 0))
        for p in preds
    )

    peak_kv_positions = max(
        (
            int(p.get("peak_kv_positions", 0))
            for p in preds
        ),
        default=0,
    )

    if torch.cuda.is_available():
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )
    else:
        peak_gpu_memory_mb = 0.0

    logical_kv_payload_mb_total = (
        logical_kv_payload_bytes_total
        / (1024 ** 2)
    )

    # Load results in JSON format
    print(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "generate_bs": args.generate_bs,
                "temperature": args.temperature,
                "latent_steps": args.latent_steps,

                "accuracy": acc,
                "correct": correct,

                "output_tokens_total": output_tokens_total,
                "output_tokens_per_sample": round(
                    output_tokens_total / n_samples, 4
                ) if n_samples else 0.0,

                "text_comm_tokens_total":
                    text_comm_tokens_total,
                "text_comm_tokens_per_sample": round(
                    text_comm_tokens_total / n_samples, 4
                ) if n_samples else 0.0,

                "kv_positions_handoff_total":
                    kv_positions_handoff_total,
                "kv_positions_handoff_per_sample": round(
                    kv_positions_handoff_total / n_samples, 4
                ) if n_samples else 0.0,

                "peak_kv_positions": peak_kv_positions,

                "logical_kv_payload_mb_total": round(
                    logical_kv_payload_mb_total, 4
                ),
                "logical_kv_payload_mb_per_sample": round(
                    logical_kv_payload_mb_total / n_samples, 4
                ) if n_samples else 0.0,

                "peak_gpu_memory_mb": round(
                    peak_gpu_memory_mb, 4
                ),

                "total_time_sec": round(total_time, 4),
                "time_per_sample_sec": round(
                    total_time / n_samples, 4
                ) if n_samples else 0.0,
            },
            ensure_ascii=False,
        )
    )



if __name__ == "__main__":
    main()
