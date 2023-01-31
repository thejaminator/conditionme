from typing import List, Optional

import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel


def complete_text_with_reward(
    prompt: str,
    target_reward: float,
    tokenizer: PreTrainedTokenizerBase,
    model: ModifiedGPT2LMHeadModel,
    temperature: float = 1.0,
) -> str:
    device = model.device
    prompt_with_bos = f"{tokenizer.bos_token}{prompt}"
    query_tensor = tokenizer.encode(prompt_with_bos, return_tensors="pt").to(device)  # type: ignore
    generation_output = model.generate(
        query_tensor,
        target_reward=target_reward,
        return_dict_in_generate=True,
        output_scores=True,
        temperature=temperature,
    )
    # generated sequence
    generated_sequence = generation_output["sequences"]
    flattened_sequence = generated_sequence.flatten()
    # convert to text
    generated_text: str = tokenizer.decode(flattened_sequence)
    return generated_text


def complete_text_with_reward_batched(
    prompt: List[str],
    target_reward: List[float],
    tokenizer: PreTrainedTokenizerBase,
    model: ModifiedGPT2LMHeadModel,
    temperature: float = 1.0,
) -> List[str]:
    device = model.device
    # for each prompt, add the bos token
    prompt_with_bos = [f"{tokenizer.bos_token}{p}" for p in prompt]
    prompt_encoding: BatchEncoding = tokenizer.batch_encode_plus(prompt_with_bos, return_tensors="pt", padding=True, return_attention_mask=True).to(device)  # type: ignore
    input_ids: torch.Tensor = prompt_encoding["input_ids"]
    attention_mask: torch.Tensor = prompt_encoding["attention_mask"]
    generation_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # convert to tensor
        target_reward=torch.tensor(target_reward).to(device),
        return_dict_in_generate=True,
        output_scores=True,
        temperature=temperature,
    )
    # generated sequence
    generated_sequence = generation_output["sequences"]
    # convert to text
    generated_text: List[str] = tokenizer.batch_decode(generated_sequence)
    return generated_text
