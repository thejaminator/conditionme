from typing import List

import torch
import typer
from datasets import Dataset, load_dataset
from transformers import GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel
from conditionme.rollout.rollout_model import (
    PromptCompletion,
    complete_text_with_reward_batched,
)
from examples.imdb.imdb_reward_model import ImdbRewardModel
from examples.imdb.train_imdb import (
    preprocessed_dataset_path,
    try_load_preprocessed_dataset,
)
from conditionme.cond_gpt2_tokenize import batch_tokenize_gpt2


def main(save_dir: str = "gdrive/My Drive/conditionme"):
    device: torch.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    # Load the model using the device
    model = ModifiedGPT2LMHeadModel.from_pretrained(save_dir).to(device)
    sentiment_reward = ImdbRewardModel(device=device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")

    dataset_tokenized: Dataset = try_load_preprocessed_dataset() or load_dataset(  # type: ignore [assignment]
        "imdb"
    ).map(
        # batched
        lambda examples: {
            "target_reward": sentiment_reward.reward_batch(
                examples["text"], batch_size=32
            )
        },
        batched=True,
    ).map(
        lambda x: batch_tokenize_gpt2(
            text=x["text"], target_rewards=x["target_reward"], tokenizer=tokenizer
        ),
        batched=True,
    )
    dataset_tokenized.save_to_disk(preprocessed_dataset_path)
    dataset_tokenized.set_format(
        type="torch", columns=["input_ids", "target_reward", "labels"]
    )

    # convert into a list of space separated tokens
    test_text_tokenized: List[List[str]] = [
        text.split(" ") for text in dataset_tokenized["test"]["text"]  # type: ignore [call-overload]
    ]
    # take the first 3 tokens from each list
    first_3_tokens_list: List[List[str]] = [text[:3] for text in test_text_tokenized]
    # join the first 3 tokens into a string
    first_3_tokens: List[str] = [" ".join(text) for text in first_3_tokens_list]
    # complete the text
    high_reward_completions: List[PromptCompletion] = complete_text_with_reward_batched(
        prompt=first_3_tokens,
        model=model,
        tokenizer=tokenizer,
        target_reward=[1.0] * len(first_3_tokens),
    )
    low_reward_completions: List[PromptCompletion] = complete_text_with_reward_batched(
        prompt=first_3_tokens,
        model=model,
        tokenizer=tokenizer,
        target_reward=[0.0] * len(first_3_tokens),
    )

    # Use the reward model to compute the actual reward of the completions
    high_reward_completions_reward: list[float] = sentiment_reward.reward_batch(
        [completion.prompt_completion for completion in high_reward_completions]
    )
    low_reward_completions_reward: list[float] = sentiment_reward.reward_batch(
        [completion.prompt_completion for completion in low_reward_completions]
    )
    # print the average
    print(
        f"Average reward of high reward completions: {sum(high_reward_completions_reward) / len(high_reward_completions_reward)}"
    )
    print(
        f"Average reward of low reward completions: {sum(low_reward_completions_reward) / len(low_reward_completions_reward)}"
    )


if __name__ == "__main__":
    # run with
    # export PYTHONPATH=.; python examples/imdb/test_imdb.py
    typer.run(main)
