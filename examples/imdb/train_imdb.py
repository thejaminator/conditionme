"""
Trains GPT2 on IMDB dataset
"""
from typing import List, Optional
import typer

import torch
from datasets import load_dataset, Dataset
from datasets.formatting.formatting import LazyBatch
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    GPT2LMHeadModel,
)

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel
from conditionme.rollout.rollout_model import (
    complete_text_with_reward_batched,
    PromptCompletion,
)
from examples.imdb.imdb_reward_model import ImdbRewardModel


def tokenize_imdb(examples: LazyBatch, eos_token: str, tokenizer) -> BatchEncoding:
    # add the eos token to the start of all text
    # the `forward` method of ModifiedGPT2LMHeadModel will modify the embedding of the eos token
    # to include the reward
    # add eos token to the end as well
    new_text = [eos_token + text + eos_token for text in examples["text"]]
    tokenizer_result = tokenizer(new_text, truncation=True, padding="longest")
    # add the precomputed reward to the result
    tokenizer_result["target_reward"] = examples["target_reward"]
    tokenizer_result["labels"] = tokenizer_result["input_ids"].copy()
    return tokenizer_result


preprocessed_dataset_path = "dataset_tokenized_imdb.hf"


def try_load_preprocessed_dataset() -> Optional[Dataset]:
    try:
        dataset_tokenized = Dataset.from_file(preprocessed_dataset_path)
        return dataset_tokenized
    except FileNotFoundError:
        return None


def main(batch_size: int, save_dir: str = "gdrive/My Drive/conditionme"):
    # Optionally save to drive
    # from google.colab import drive
    # drive.mount('/content/gdrive')

    # Download and tokenize the dataset
    imdb_dataset = load_dataset("imdb")
    # limit the dataset to 100 examples
    # compute the reward for each example
    # prefer gpu if available
    device: torch.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    sentiment_reward = ImdbRewardModel(device=device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos_token: str = tokenizer.eos_token
    # see https://github.com/huggingface/transformers/issues/2630
    tokenizer.pad_token = tokenizer.eos_token
    dataset_tokenized: Dataset = try_load_preprocessed_dataset() or imdb_dataset.map(  # type: ignore
        # batched
        lambda examples: {
            "target_reward": sentiment_reward.reward_batch(
                examples["text"], batch_size=32
            )
        },
        batched=True,
    ).map(
        lambda x: tokenize_imdb(x, eos_token, tokenizer),
        batched=True,
    )
    dataset_tokenized.save_to_disk(preprocessed_dataset_path)
    dataset_tokenized.set_format(
        type="torch", columns=["input_ids", "target_reward", "labels"]
    )

    print("ok")

    # Train the model using the device
    gpt2_model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained("gpt2").to(
        device
    )
    model = ModifiedGPT2LMHeadModel(existing_head_model=gpt2_model)

    training_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized["train"],
        tokenizer=tokenizer,
    )
    trainer.train()

    # Save the model
    trainer.save_model(save_dir)

    # convert into a list of space separated tokens
    test_text_tokenized: List[List[str]] = [
        text.split(" ") for text in dataset_tokenized["test"]["text"]  # type: ignore
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
    # export PYTHONPATH=.; python examples/imdb/train_imdb.py
    typer.run(main)
