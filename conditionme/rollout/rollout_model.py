import dataclasses
from typing import List

import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel


@dataclasses.dataclass
class PromptCompletion:
    prompt: str
    completion: str

    @property
    def prompt_completion(self) -> str:
        return f"{self.prompt}{self.completion}"


def complete_text_with_reward(
    prompt: str,
    target_reward: float,
    tokenizer: PreTrainedTokenizerBase,
    model: ModifiedGPT2LMHeadModel,
    temperature: float = 1.0,
) -> str:
    device: torch.device = model.device  # type: ignore
    prompt_with_bos = f"{tokenizer.bos_token}{prompt}"
    query_tensor: torch.tensor = tokenizer.encode(prompt_with_bos, return_tensors="pt").to(device)  # type: ignore
    generation_output = model.generate(  # type: ignore
        inputs=query_tensor,
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
) -> List[PromptCompletion]:
    device: torch.device = model.device  # type: ignore
    # for each prompt, add the bos token
    prompt_with_bos = [f"{tokenizer.bos_token}{p}" for p in prompt]
    prompt_encoding: BatchEncoding = tokenizer.batch_encode_plus(prompt_with_bos, return_tensors="pt", padding=True, return_attention_mask=True).to(device)  # type: ignore
    input_ids: torch.Tensor = prompt_encoding["input_ids"]
    attention_mask: torch.Tensor = prompt_encoding["attention_mask"]
    generation_output = model.generate(  # type: ignore
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
    generated_text: List[str] = tokenizer.batch_decode(
        generated_sequence, skip_special_tokens=True
    )
    # split the generated text into the prompt and the completion
    prompt_completion: List[PromptCompletion] = []
    for i, _prompt in enumerate(prompt):
        completion = generated_text[i].lstrip(_prompt)
        prompt_completion.append(
            PromptCompletion(prompt=_prompt, completion=completion)
        )
    return prompt_completion
