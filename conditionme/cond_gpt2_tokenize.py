from copy import copy
from typing import Sequence, List

import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding, TensorType

from settings import DEFAULT_REWARD_TOKEN_ID


def set_up_decoder_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedTokenizerBase:
    # shallow copy tokenizer to avoid unexpected side effects
    new_tokenizer: PreTrainedTokenizerBase = copy(tokenizer)
    # need to manually set the pad token to the eos token
    new_tokenizer.pad_token = new_tokenizer.eos_token
    new_tokenizer.pad_token_id = new_tokenizer.eos_token_id
    # since this is a decoder, we need to left pad to work with HF generate
    # https://github.com/huggingface/transformers/issues/3021#issuecomment-1231526631
    new_tokenizer.padding_side = "left"
    return new_tokenizer


def batch_tokenize_gpt2(
    text: Sequence[str],
    target_rewards: Sequence[float],
    tokenizer: PreTrainedTokenizerBase,
    add_eos_at_end: bool,
    # NOTE: Avoid using special tokens like eos_token, bos_token, because those may get masked automatically by data_collator
    reward_token_id: int = DEFAULT_REWARD_TOKEN_ID,
) -> BatchEncoding:
    # TODO: Do the padding in the data collator instead?
    # shallow copy tokenizer to avoid unexpected side effects
    new_tokenizer: PreTrainedTokenizerBase = set_up_decoder_tokenizer(tokenizer)
    assert len(text) == len(target_rewards)
    # add the reward token to the start of all text, before we apply the padding
    # the `forward` method of ModifiedGPT2LMHeadModel will modify the embedding of the reward_token using the position provided
    reward_token = new_tokenizer.decode([reward_token_id])
    maybe_eos: str = new_tokenizer.eos_token if add_eos_at_end else ""
    new_text = [reward_token + t + maybe_eos for t in text]
    tokenizer_result = new_tokenizer(new_text, truncation=True, padding="longest")
    inputs_ids = tokenizer_result["input_ids"]
    for i, input_ids in enumerate(inputs_ids):
        assert reward_token_id in input_ids
    # BatchEncoding will have "input_ids", "attention_mask, "target_reward", "labels", "target_reward_position"
    # add the precomputed reward to the result
    tokenizer_result["target_reward"] = target_rewards
    attention_mask = tokenizer_result["attention_mask"]
    # If the token is masked, set the label to -100 so it is ignored in the loss function
    labels: List[List[int]] = []
    for mask in attention_mask:
        labels.append([-100 if x == 0 else x for x in mask])
    tokenizer_result["labels"] = labels
    # convert to tensors
    new_dict = BatchEncoding(tensor_type=TensorType.PYTORCH)
    for key in tokenizer_result:
        new_dict[key] = torch.tensor(tokenizer_result[key])
    return new_dict

