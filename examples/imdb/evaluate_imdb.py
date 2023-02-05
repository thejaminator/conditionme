from pathlib import Path
from typing import List

import numpy as np
import torch
import typer
from datasets import Dataset, load_dataset
from transformers import GPT2LMHeadModel, AutoTokenizer

from conditionme.cond_gpt2_tokenize import batch_tokenize_gpt2
from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel
from conditionme.normalization.normalizer import RewardNormalizer
from conditionme.rollout.rollout_model import (
    PromptCompletion,
    complete_text_with_reward_batched,
)
from conditionme.statistics.calculate_distribution import (
    calculate_distribution_statistics,
)
from conditionme.statistics.create_reward_table import (
    reward_evaluation_rows,
    reward_evaluation_table,
)
from conditionme.statistics.plotting import plot_scatterplot_and_correlation
from examples.imdb.imdb_reward_model import ImdbRewardModel
from examples.imdb.reload_dataset import (
    try_load_preprocessed_dataset,
)


def evaluate_test_set(
    test_text: List[str],
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    sentiment_reward: ImdbRewardModel,
    normalizer: RewardNormalizer,
    limit: int,
):
    # convert into a list of space separated tokens
    test_text_tokenized: List[List[str]] = [
        text.split(" ") for text in test_text[:limit]
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
    high_target_actual_reward: List[float] = sentiment_reward.reward_batch(
        [completion.prompt_completion for completion in high_reward_completions]
    )
    low_target_actual_reward: List[float] = sentiment_reward.reward_batch(
        [completion.prompt_completion for completion in low_reward_completions]
    )
    print(sum(high_target_actual_reward))
    print(sum(low_target_actual_reward))
    # print the stats
    # log training target_reward
    high_reward_dist = calculate_distribution_statistics(dist=high_target_actual_reward)
    low_reward_dist = calculate_distribution_statistics(dist=low_target_actual_reward)
    print(f"High reward distribution: {high_reward_dist}")
    print(f"Low reward distribution: {low_reward_dist}")

    # create csv of rewards
    high_reward_rows = reward_evaluation_rows(
        prompt_completions=high_reward_completions,
        target_rewards=normalizer.normalize_rewards(
            [1.0] * len(high_reward_completions)
        ),
        actual_rewards=high_target_actual_reward,
    )
    reward_evaluation_table(high_reward_rows).to_csv(
        "high_reward_completions.csv", index=False
    )

    low_reward_rows = reward_evaluation_rows(
        prompt_completions=low_reward_completions,
        target_rewards=normalizer.normalize_rewards(
            [0.0] * len(high_reward_completions)
        ),
        actual_rewards=low_target_actual_reward,
    )
    reward_evaluation_table(low_reward_rows).to_csv(
        "low_reward_completions.csv", index=False
    )

    # Randomly sample target rewards to plot correlation graph
    # Sample from a uniform distribution of 0 to 1 since those are the lower and upper bounds of the target rewards
    sampled_target_rewards = np.random.uniform(0, 1, size=len(first_3_tokens))
    # Use the normalizer to normalize the target rewards
    normalized_target_rewards = normalizer.normalize_rewards(
        sampled_target_rewards.tolist()
    )
    # complete the text
    sampled_completions: List[PromptCompletion] = complete_text_with_reward_batched(
        prompt=first_3_tokens,
        model=model,
        tokenizer=tokenizer,
        target_reward=normalized_target_rewards,
    )
    # Use the reward model to compute the actual reward of the completions
    sampled_actual_rewards: List[float] = sentiment_reward.reward_batch(
        [completion.prompt_completion for completion in sampled_completions]
    )
    # create csv of rewards
    sampled_rows = reward_evaluation_rows(
        prompt_completions=sampled_completions,
        target_rewards=normalized_target_rewards,
        actual_rewards=sampled_actual_rewards,
    )
    reward_evaluation_table(sampled_rows).to_csv("sampled_completions.csv", index=False)
    # Plot the correlation graph
    plot_results = plot_scatterplot_and_correlation(
        x=sampled_target_rewards.tolist(),
        y=sampled_actual_rewards,
        title="Correlation between target and actual reward",
        xlabel="Target reward",
        ylabel="Actual reward",
    )
    plot_results.figure.savefig("correlation.png")


def main(save_dir: str = "gdrive/My Drive/conditionme", limit: int = 1000):
    device: torch.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    # Load the normalizer
    normalizer = RewardNormalizer.load_normalizer(Path(save_dir))
    print(f"Loading model from {save_dir}")
    # Load the model using the device
    model = ModifiedGPT2LMHeadModel.from_pretrained(save_dir).to(device)
    sentiment_reward = ImdbRewardModel(device=device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    print("Loading dataset")

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
            text=x["text"],
            target_rewards=x["target_reward"],
            tokenizer=tokenizer,
            add_eos_at_end=True,
        ),
        batched=True,
    )
    dataset_tokenized.set_format(
        type="torch", columns=["input_ids", "target_reward", "attention_mask"]
    )
    test_text: List[str] = dataset_tokenized["test"]["text"]  # type: ignore [call-overload]
    evaluate_test_set(
        test_text=test_text,
        model=model,
        tokenizer=tokenizer,
        sentiment_reward=sentiment_reward,
        limit=limit,
        normalizer=normalizer,
    )


if __name__ == "__main__":
    # run with
    # export PYTHONPATH=.; python examples/imdb/evaluate_imdb.py --save-dir saved/
    typer.run(main)
