from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import torch
import argparse
from vllm import SamplingParams
import pdb
import math
from selectkv.selector import HybridKVSelector

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class SelectKVMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=args.max_new_tokens,
        )
        self.task = args.task

        # SelectKV configuration
        self.selectkv_budget_ratio = float(
            getattr(args, "selectkv_budget_ratio", 0.80)
        )
        self.selectkv_recent_tokens = int(
            getattr(args, "selectkv_recent_tokens", 4)
        )
        self.selectkv_overlap_pool_fraction = float(
            getattr(args, "selectkv_overlap_pool_fraction", 0.50)
        )

        if not 0.0 < self.selectkv_budget_ratio <= 1.0:
            raise ValueError("selectkv_budget_ratio must be in (0, 1].")

        self.selectkv_selector = HybridKVSelector(
            overlap_pool_fraction=self.selectkv_overlap_pool_fraction
        )

        # Diagnostics accumulated across a run.
        self.selectkv_stats = {
            "selection_events": 0,
            "kv_positions_before": 0,
            "kv_positions_after": 0,
        }

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @staticmethod
    def _cache_to_legacy(past_kv):
        if past_kv is None:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            return past_kv.to_legacy_cache()
        return past_kv

    @staticmethod
    def _restore_cache_type(original_cache, legacy_cache):
        if (
            Cache is not None
            and isinstance(original_cache, Cache)
            and hasattr(original_cache.__class__, "from_legacy_cache")
        ):
            return original_cache.__class__.from_legacy_cache(legacy_cache)
        return legacy_cache

    @staticmethod
    def _slice_cache_indices(past_kv, selected_indices):
        """Apply identical token indices to K and V tensors in every layer."""
        if past_kv is None:
            return None

        original_cache = past_kv

        if Cache is not None and isinstance(past_kv, Cache):
            past_kv = past_kv.to_legacy_cache()

        selected_indices = sorted(int(i) for i in selected_indices)

        trimmed_layers = []

        for layer in past_kv:
            if not isinstance(layer, (tuple, list)):
                trimmed_layers.append(layer)
                continue

            trimmed = []

            for tensor in layer:
                if not torch.is_tensor(tensor) or tensor.dim() < 3:
                    trimmed.append(tensor)
                    continue

                seq_len = tensor.shape[-2]

                valid = [
                    i for i in selected_indices
                    if 0 <= i < seq_len
                ]

                if not valid:
                    trimmed.append(tensor[..., 0:0, :].contiguous())
                    continue

                index = torch.tensor(
                    valid,
                    dtype=torch.long,
                    device=tensor.device,
                )

                trimmed.append(
                    torch.index_select(
                        tensor,
                        dim=-2,
                        index=index,
                    ).contiguous()
                )

            trimmed_layers.append(tuple(trimmed))

        trimmed_layers = tuple(trimmed_layers)

        return SelectKVMASMethod._restore_cache_type(
            original_cache,
            trimmed_layers,
        )

    @staticmethod
    def _compute_attention_persistence(
        all_steps_attentions,
        seq_len,
    ):
        """
        Aggregate attention received by each cached position across
        latent reasoning steps, layers, batches, and heads.
        """
        if seq_len <= 0:
            return []

        scores = torch.zeros(
            seq_len,
            dtype=torch.float32,
        )

        counts = torch.zeros(
            seq_len,
            dtype=torch.float32,
        )

        for step_attentions in all_steps_attentions or []:
            for attention in step_attentions or []:

                if attention is None or attention.dim() < 4:
                    continue

                # [B, H, Q, K] -> attention from newest query.
                att = attention[..., -1, :].detach().float().cpu()

                # Average across batch and heads.
                att = att.mean(dim=(0, 1))

                usable = min(seq_len, int(att.shape[-1]))

                if usable <= 0:
                    continue

                scores[:usable] += att[:usable]
                counts[:usable] += 1.0

        counts = torch.clamp(counts, min=1.0)

        return (scores / counts).tolist()

    @staticmethod
    def _compute_receiver_relevance(
        past_kv,
        receiver_hidden,
    ):
        """
        Compute receiver-conditioned semantic relevance.

        Cached keys are averaged across layers/heads and compared
        against the receiver representation using cosine similarity.
        """
        if past_kv is None:
            return []

        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
        else:
            legacy = past_kv

        key_layers = []

        for layer in legacy:
            if not isinstance(layer, (tuple, list)) or len(layer) == 0:
                continue

            key = layer[0]

            if not torch.is_tensor(key) or key.dim() != 4:
                continue

            # [B, H, S, D] -> [S, D]
            key_mean = key.detach().float().mean(dim=(0, 1))
            key_layers.append(key_mean)

        if not key_layers:
            return []

        min_seq = min(k.shape[-2] for k in key_layers)
        min_dim = min(k.shape[-1] for k in key_layers)

        keys = torch.stack(
            [
                k[:min_seq, :min_dim]
                for k in key_layers
            ],
            dim=0,
        ).mean(dim=0)

        query = receiver_hidden.detach().float()

        if query.dim() > 1:
            query = query.mean(dim=0)

        query = query.flatten()

        # Qwen hidden dimension is larger than one attention head.
        # Fold it into head-sized chunks when possible.
        if query.numel() != min_dim:
            if query.numel() % min_dim == 0:
                query = query.view(-1, min_dim).mean(dim=0)
            else:
                query = query[:min_dim]

        keys = torch.nn.functional.normalize(
            keys,
            p=2,
            dim=-1,
        )

        query = torch.nn.functional.normalize(
            query,
            p=2,
            dim=-1,
        )

        scores = torch.matmul(keys, query)

        return scores.detach().cpu().tolist()

    def _apply_selectkv(
        self,
        past_kv,
        receiver_query,
        step_attentions,
    ):
        """Select and physically trim the inter-agent KV cache."""

        seq_len = _past_length(past_kv)

        if seq_len <= 1:
            return past_kv, {
                "kv_positions_before": seq_len,
                "kv_positions_after": seq_len,
                "retention_ratio": 1.0,
            }

        relevance_scores = self._compute_receiver_relevance(
            past_kv,
            receiver_query,
        )

        persistence_scores = self._compute_attention_persistence(
            step_attentions,
            seq_len,
        )

        n = min(
            seq_len,
            len(relevance_scores),
            len(persistence_scores),
        )

        if n <= 0:
            return past_kv, {
                "kv_positions_before": seq_len,
                "kv_positions_after": seq_len,
                "retention_ratio": 1.0,
            }

        relevance_scores = relevance_scores[:n]
        persistence_scores = persistence_scores[:n]

        budget = max(
            1,
            int(math.ceil(n * self.selectkv_budget_ratio)),
        )

        recent_count = min(
            self.selectkv_recent_tokens,
            budget,
            n,
        )

        protected_indices = list(
            range(n - recent_count, n)
        )

        result = self.selectkv_selector.select(
            relevance_scores=relevance_scores,
            persistence_scores=persistence_scores,
            budget=budget,
            protected_indices=protected_indices,
        )

        self.selectkv_stats["selection_events"] += 1
        self.selectkv_stats["kv_positions_before"] += n
        self.selectkv_stats["kv_positions_after"] += len(
            result.selected_indices
        )

        trimmed_kv = self._slice_cache_indices(
            past_kv,
            result.selected_indices,
        )

        selectkv_info = {
            "kv_positions_before": n,
            "kv_positions_after": len(result.selected_indices),
            "retention_ratio": (
                len(result.selected_indices) / n
                if n > 0 else 1.0
            ),
            "budget": result.budget,
            "overlap_count": len(result.overlap_indices),
            "relevance_only_count": len(result.relevance_only),
            "persistence_only_count": len(result.persistence_only),
            "protected_count": len(result.protected_indices),
        }

        return trimmed_kv, selectkv_info

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        for agent in self.agents:

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]


            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # SelectKV requires both sender attention history and
                # hidden states in addition to the resulting KV cache.
                past_kv, hidden_states_dict, step_attentions = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                    return_attentions=True,
                    return_hidden_states=True,
                )

                # Receiver-conditioned query representation.
                # final_hidden summarizes the sender trajectory and is used
                # to score which cache positions remain relevant downstream.
                receiver_query = hidden_states_dict["final_hidden"]

                # Select the communication subset before handing the cache
                # to the next agent.
                past_kv, selectkv_info = self._apply_selectkv(
                    past_kv,
                    receiver_query=receiver_query,
                    step_attentions=step_attentions,
                )

                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:

                past_for_decoding = past_kv if self.latent_steps > 0 else None

                if self.args.think:
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
            
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results
    
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        for agent in self.agents:
            
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
                
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                
                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ) 
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).squeeze(0).to(self.vllm_device)
                
                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position    
                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    # Get the text up to and including "<|im_start|>user\n"
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                    
                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency
                    
                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                # Pad back to max length if needed
                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([
                    torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                    for x in whole_prompt_emb_list
                ])

                # else:
                    # Get full prompt embedding from cat with previous ones 
                    # B L H B L H
                    # whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)
                
                # pdb.set_trace()              
                
                # Use vLLM 
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb 
                ]
                
                
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]
                    
                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )


        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
