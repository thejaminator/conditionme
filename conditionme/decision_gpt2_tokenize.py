from copy import copy
from typing import Sequence, List, NewType

import torch
from transformers import (
    PreTrainedTokenizerBase,
    BatchEncoding,
    TensorType,
)
from transformers.utils import PaddingStrategy

# This is a new type - it isn't enforced at runtime, but it is enforced at type-checking time
DecisionTokenizer = NewType("DecisionTokenizer", PreTrainedTokenizerBase)  # type: ignore [valid-newtype]


def create_decision_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> DecisionTokenizer:
    # shallow copy tokenizer to avoid unexpected side effects
    new_tokenizer: PreTrainedTokenizerBase = copy(tokenizer)
    # minus one to the sequence length to account for the reward token
    new_tokenizer.model_max_length = new_tokenizer.model_max_length - 1
    # need to manually set the pad token to the eos token
    new_tokenizer.pad_token = new_tokenizer.eos_token
    new_tokenizer.pad_token_id = new_tokenizer.eos_token_id
    # since this is a decoder, we need to left pad to work with HF generate
    # https://github.com/huggingface/transformers/issues/3021#issuecomment-1231526631
    new_tokenizer.padding_side = "left"
    # we also need to truncate from the left
    new_tokenizer.truncation_side = "left"
    return DecisionTokenizer(new_tokenizer)


def batch_tokenize_gpt2(
    text: Sequence[str],
    target_rewards: Sequence[float],
    decision_tokenizer: DecisionTokenizer,
    add_eos_at_end: bool,
) -> BatchEncoding:
    assert len(text) == len(target_rewards)

    tokenizer_result = decision_tokenizer.__call__(
        text,
        padding=True,
        truncation=True,
        max_length=decision_tokenizer.model_max_length,
        add_special_tokens=True if add_eos_at_end else False,
    )

    # BatchEncoding will have "input_ids", "attention_mask, "target_reward", "labels"
    # add target_reward to the result
    tokenizer_result["target_reward"] = target_rewards
    # convert to tensors
    new_dict = BatchEncoding(tensor_type=TensorType.PYTORCH)
    for key in tokenizer_result:
        new_dict[key] = torch.tensor(tokenizer_result[key])
    return new_dict
