"""
Trains GPT2 on IMDB dataset
"""
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import torch
import typer
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BatchEncoding,
)

from conditionme.cond_gpt2_tokenize import batch_tokenize_gpt2, set_up_decoder_tokenizer
from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel
from conditionme.normalization.normalizer import (
    RewardNormalizer,
    StandardScaleNormalizer,
    DoNothingNormalizer,
)
from conditionme.statistics.calculate_distribution import (
    calculate_distribution_statistics,
)
from examples.imdb.evaluate_imdb import evaluate_test_set
from examples.imdb.imdb_reward_model import ImdbRewardModel
from examples.imdb.reload_dataset import (
    preprocessed_dataset_path,
    try_load_preprocessed_dataset,
)


class GPT2ModelOptions(Enum):
    # see https://huggingface.co/transformers/v2.2.0/pretrained_models.html
    gpt2 = "gpt2"
    gpt2_medium = "gpt2-medium"
    gpt2_large = "gpt2-large"
    gpt2_xl = "gpt2-xl"


def batch_normalize(
    batch: BatchEncoding,
    normalizer: RewardNormalizer,
) -> BatchEncoding:
    # normalize the reward
    normalized_reward = normalizer.normalize_rewards(batch["target_reward"])
    # replace the reward with the normalized reward
    batch["target_reward"] = normalized_reward
    return batch


def train_imdb(
    batch_size: int,
    epochs: int,
    save_dir: str,
    tokenizer: AutoTokenizer,
    gpt2_model: ModifiedGPT2LMHeadModel,
    learning_rate: float,
    # must contain "train", "test", and "text" keys
    dataset: Union[DatasetDict, Dataset],
    reward_model: ImdbRewardModel,
) -> None:

    dataset_tokenized: Dataset = dataset.map(  # type: ignore
        # batched
        lambda examples: {
            "target_reward": reward_model.reward_batch(examples["text"], batch_size=32)
        },
        batched=True,
    ).map(
        lambda x: batch_tokenize_gpt2(
            x["text"],
            target_rewards=x["target_reward"],
            tokenizer=tokenizer,
            add_eos_at_end=True,
        ),
        batch_size=batch_size,  # We don't have to pad so much if batch_size is smaller
        batched=True,
    )
    normalizer: RewardNormalizer = DoNothingNormalizer.from_rewards(
        rewards=dataset_tokenized["train"]["target_reward"]  # type: ignore
    )
    # update the dataset with the normalized rewards
    normalized_dataset = dataset_tokenized.map(
        lambda x: batch_normalize(x, normalizer=normalizer), batched=True
    )

    # Save the normalizer
    normalizer.save_normalizer(Path(save_dir))

    # log training target_reward
    training_reward_dist = calculate_distribution_statistics(
        dist=normalized_dataset["train"]["target_reward"]  # type: ignore
    )
    print(f"Training target_reward distribution: {training_reward_dist}")
    normalized_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "target_reward",
            "attention_mask",
        ],
    )

    print("ok")

    training_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        # default transformer package is  5e-5
        learning_rate=learning_rate,
    )
    conditional_tokenizer = set_up_decoder_tokenizer(tokenizer)
    trainer = Trainer(
        model=gpt2_model,
        args=training_args,
        train_dataset=normalized_dataset["train"],
        tokenizer=conditional_tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=conditional_tokenizer, mlm=False
        ),
    )
    trainer.train()

    # Save the model
    gpt2_model.save_pretrained(save_dir)
    test_text: List[str] = normalized_dataset["test"]["text"]  # type: ignore [call-overload]
    evaluate_test_set(
        test_text=test_text,
        model=gpt2_model,
        tokenizer=tokenizer,
        sentiment_reward=reward_model,
        limit=1000,
        normalizer=normalizer,
        save_dir=Path(save_dir),
    )


def main(
    batch_size: int = 1,
    epochs: int = 1,
    save_dir: str = "gdrive/My Drive/conditionme",
    model: GPT2ModelOptions = GPT2ModelOptions.gpt2,
    learning_rate: float = 5e-5,
    device: Optional[str] = None,
):
    # Optionally save to drive
    # from google.colab import drive
    # drive.mount('/content/gdrive')

    # Download and tokenize the dataset
    imdb_dataset: Dataset = load_dataset("imdb")  # type: ignore
    # limit the dataset to 100 examples
    # compute the reward for each example
    # prefer gpu if available
    device_selected: torch.device = (
        torch.device(device)
        if device
        else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    )
    sentiment_reward = ImdbRewardModel(device=device_selected)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    gpt2_model = ModifiedGPT2LMHeadModel.from_pretrained(model.value).to(device)
    train_imdb(
        batch_size=batch_size,
        epochs=epochs,
        save_dir=save_dir,
        tokenizer=tokenizer,
        gpt2_model=gpt2_model,
        learning_rate=learning_rate,
        dataset=imdb_dataset,
        reward_model=sentiment_reward,
    )


if __name__ == "__main__":
    # run with
    # export PYTHONPATH=.; python examples/imdb/train_imdb.py --batch-size 10 --epochs 1
    # export PYTHONPATH=.; python examples/imdb/train_imdb.py --batch-size 1 --epochs 1 --model gpt2-medium --save-dir saved_medium
    typer.run(main)
