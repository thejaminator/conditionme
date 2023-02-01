from copy import copy
from typing import Sequence, List

import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding, TensorType


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
) -> BatchEncoding:
    # shallow copy tokenizer to avoid unexpected side effects
    new_tokenizer: PreTrainedTokenizerBase = set_up_decoder_tokenizer(tokenizer)
    assert len(text) == len(target_rewards)
    # add the reward token to the start of all text, before we apply the padding
    reward_token = new_tokenizer.eos_token
    # the `forward` method of ModifiedGPT2LMHeadModel will modify the embedding of the reward_token using the position provided
    maybe_eos: str = new_tokenizer.eos_token if add_eos_at_end else ""
    new_text = [reward_token + t + maybe_eos for t in text]
    tokenizer_result = new_tokenizer(new_text, truncation=True, padding="longest")
    # BatchEncoding will have "input_ids", "attention_mask, "target_reward", "labels", "target_reward_position"
    # add the precomputed reward to the result
    tokenizer_result["target_reward"] = target_rewards
    tokenizer_result["labels"] = tokenizer_result["input_ids"].copy()
    attention_mask = tokenizer_result["attention_mask"]
    target_reward_position: List[int] = [x.index(1) for x in attention_mask]
    tokenizer_result["target_reward_position"] = target_reward_position
    # convert to tensors
    new_dict = BatchEncoding(tensor_type=TensorType.PYTORCH)
    for key in tokenizer_result:
        new_dict[key] = torch.tensor(tokenizer_result[key])
    return new_dict


def test_batch_tokenize_gpt2_reward_position():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    text = ["hello hello", "hello hello hello"]
    target_rewards = [1.0, 1.0]
    tokenizer_result = batch_tokenize_gpt2(
        text=text,
        target_rewards=target_rewards,
        tokenizer=tokenizer,
        add_eos_at_end=True,
    )
    # the first text has two tokens, the second has three
    # so the reward token should be at position 1 for the first, and 0 for the second, after padding
    assert tokenizer_result["target_reward_position"][0] == 1
    assert tokenizer_result["target_reward_position"][1] == 0
