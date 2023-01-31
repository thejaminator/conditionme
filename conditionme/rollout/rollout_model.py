from typing import List, Optional

from transformers import GenerationMixin, PreTrainedTokenizer, PreTrainedTokenizerBase

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel


def complete_text_with_reward(
    prompt: str,
    target_reward: float,
    tokenizer: PreTrainedTokenizerBase,
    model: ModifiedGPT2LMHeadModel,
    temperature: float = 1.0,
    # max int
    max_length: int = 9999999999,
) -> str:
    device = model.device
    query_tensor = tokenizer.encode(prompt, return_tensors="pt").to(device)  # type: ignore
    generation_output = model.generate(
        query_tensor,
        target_reward=target_reward,
        return_dict_in_generate=True,
        output_scores=True,
        temperature=temperature,
        max_length=max_length,
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
    max_length: Optional[int] = None,
) -> List[str]:
    device = model.device
    query_tensor = tokenizer.batch_encode_plus(prompt, return_tensors="pt").to(device)  # type: ignore
    generation_output = model.generate(
        query_tensor,
        target_reward=target_reward,
        return_dict_in_generate=True,
        output_scores=True,
        temperature=temperature,
        max_length=max_length,
    )
    # generated sequence
    generated_sequence = generation_output["sequences"]
    flattened_sequence = generated_sequence.flatten()
    # convert to text
    generated_text: List[str] = tokenizer.batch_decode(flattened_sequence)
    return generated_text
