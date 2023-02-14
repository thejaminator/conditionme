import dataclasses
from typing import List, Tuple

import torch
from slist import Slist
from transformers import PreTrainedTokenizerBase, BatchEncoding, GenerationConfig

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel
from conditionme.cond_gpt2_tokenize import batch_tokenize_gpt2, set_up_decoder_tokenizer


@dataclasses.dataclass
class PromptCompletion:
    prompt: str
    completion: str

    @property
    def prompt_completion(self) -> str:
        return f"{self.prompt}{self.completion}"


@dataclasses.dataclass
class PromptWithTargetReward:
    prompt: str
    target_reward: float


def complete_text_with_reward(
    prompt: str,
    target_reward: float,
    tokenizer: PreTrainedTokenizerBase,
    model: ModifiedGPT2LMHeadModel,
    temperature: float = 1.0,
    max_new_tokens: int = 100,
) -> PromptCompletion:
    return __complete_text_with_reward_batched_helper(
        prompts_and_targets=Slist([PromptWithTargetReward(prompt, target_reward)]),
        tokenizer=tokenizer,
        model=model,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    ).first_or_raise()


def complete_text_with_reward_batched(
    prompt: List[str],
    target_reward: List[float],
    tokenizer: PreTrainedTokenizerBase,
    model: ModifiedGPT2LMHeadModel,
    temperature: float = 1.0,
    batch_size: int = 4,
    max_new_tokens: int = 100,
) -> List[PromptCompletion]:
    prompts_rewards: Slist[PromptWithTargetReward] = (
        Slist(prompt)
        .zip(target_reward)
        .map(lambda p: PromptWithTargetReward(prompt=p[0], target_reward=p[1]))
    )
    grouped = prompts_rewards.grouped(batch_size)
    completions: List[PromptCompletion] = []
    for group in grouped:
        completions.extend(
            __complete_text_with_reward_batched_helper(
                prompts_and_targets=group,
                tokenizer=tokenizer,
                model=model,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        )
    return completions


def __complete_text_with_reward_batched_helper(
    prompts_and_targets: Slist[PromptWithTargetReward],
    tokenizer: PreTrainedTokenizerBase,
    model: ModifiedGPT2LMHeadModel,
    temperature: float,
    max_new_tokens: int,
) -> Slist[PromptCompletion]:
    # shallow copy the tokenizer to avoid unexpected side effects
    new_tokenizer = set_up_decoder_tokenizer(tokenizer)
    device: torch.device = model.device  # type: ignore
    # for each prompt, add the bos token which will be the reward token
    encoding = batch_tokenize_gpt2(
        text=prompts_and_targets.map(lambda x: x.prompt),
        target_rewards=prompts_and_targets.map(lambda x: x.target_reward),
        tokenizer=new_tokenizer,
        add_eos_at_end=False,
    )
    encoding.to(device)
    input_ids: torch.Tensor = encoding["input_ids"]
    attention_mask: torch.Tensor = encoding["attention_mask"]
    generation_output = model.generate(  # type: ignore
        input_ids=input_ids,
        attention_mask=attention_mask,
        # convert to tensor
        target_reward=torch.tensor(
            prompts_and_targets.map(lambda x: x.target_reward)
        ).to(device),
        return_dict_in_generate=True,
        output_scores=True,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=new_tokenizer.eos_token_id,
        generation_config=GenerationConfig(
            bos_token_id=new_tokenizer.bos_token_id,
            eos_token_id=new_tokenizer.eos_token_id,
        ),
    )
    # generated sequence
    generated_sequence = generation_output["sequences"]  # type: ignore
    # convert to text
    generated_text: List[str] = new_tokenizer.batch_decode(
        generated_sequence, skip_special_tokens=True
    )
    # split the generated text into the prompt and the completion
    prompt_completion: Slist[PromptCompletion] = Slist()
    prompts = prompts_and_targets.map(lambda x: x.prompt)
    for i, _prompt in enumerate(prompts):
        full_prompt_chars = len(_prompt)
        completion = generated_text[i][full_prompt_chars:]
        prompt_completion.append(
            PromptCompletion(prompt=_prompt, completion=completion)
        )
    return prompt_completion
