from copy import copy
from typing import Sequence, List, Optional

import torch
from transformers import (
    PreTrainedTokenizerBase,
    BatchEncoding,
    TensorType,
    GPT2Tokenizer,
)
from transformers.utils import PaddingStrategy

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
    # we also need to truncate from the left
    new_tokenizer.truncation_side = "left"
    # TODO: Make this configurable
    new_tokenizer.reward_token_id = DEFAULT_REWARD_TOKEN_ID
    new_tokenizer.reward_token = new_tokenizer.decode([DEFAULT_REWARD_TOKEN_ID])
    return new_tokenizer


def manual_keep_front_truncation(
    input_ids: List[List[int]],
    max_length: int,
) -> List[List[int]]:
    # Because the transformers package truncates by keeping the last tokens, rather than the first tokens
    # we need to manually truncate the text
    # TODO: Check whether this is really true?
    return [row[:max_length] for row in input_ids]


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
    tokenized_ids = new_tokenizer(text)["input_ids"]
    # add reward_token to the start of all text, and add eos_token to the end of all text
    tokenized_ids_with_special_tokens: List[List[int]] = [
        [reward_token_id]
        + row
        + ([new_tokenizer.eos_token_id] if add_eos_at_end else [])
        for row in tokenized_ids
    ]
    # Manually pad and truncate we want to add the token id ourselves
    tokenizer_result = new_tokenizer.pad(
        {
            "input_ids": manual_keep_front_truncation(
                tokenized_ids_with_special_tokens, max_length=tokenizer.model_max_length
            )
        },
        padding=PaddingStrategy.LONGEST,
        return_attention_mask=True,
    )
    inputs_ids = tokenizer_result["input_ids"]
    for i, input_id_row in enumerate(inputs_ids):
        if reward_token_id not in input_id_row:
            decoded_text = new_tokenizer.decode(input_id_row)
            raise ValueError(
                f"New Text: {text[i]} did not get tokenized correctly. Got tokenize to {input_id_row}\nDecoded text {decoded_text}"
            )

    # BatchEncoding will have "input_ids", "attention_mask, "target_reward", "labels"
    # add the precomputed reward to the result
    tokenizer_result["target_reward"] = target_rewards
    # convert to tensors
    new_dict = BatchEncoding(tensor_type=TensorType.PYTORCH)
    for key in tokenizer_result:
        new_dict[key] = torch.tensor(tokenizer_result[key])
    return new_dict
