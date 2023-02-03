"""
Trains GPT2 on IMDB dataset
"""
from enum import Enum
from typing import List, Optional

import torch
import typer
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
)

from conditionme.cond_gpt2_tokenize import batch_tokenize_gpt2
from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel
from conditionme.statistics.calculate_distribution import (
    calculate_distribution_statistics,
)
from examples.imdb.imdb_reward_model import ImdbRewardModel
from examples.imdb.reload_dataset import (
    preprocessed_dataset_path,
    try_load_preprocessed_dataset,
)
from examples.imdb.test_imdb import evaluate_test_set


class GPT2ModelOptions(Enum):
    # see https://huggingface.co/transformers/v2.2.0/pretrained_models.html
    gpt2 = "gpt2"
    gpt2_medium = "gpt2-medium"
    gpt2_large = "gpt2-large"
    gpt2_xl = "gpt2-xl"


def main(
    batch_size: int = 1,
    epochs: int = 1,
    save_dir: str = "gdrive/My Drive/conditionme",
    model: GPT2ModelOptions = GPT2ModelOptions.gpt2,
):
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
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    # see https://github.com/huggingface/transformers/issues/2630
    tokenizer.pad_token = tokenizer.eos_token
    cached_dataset: Optional[Dataset] = try_load_preprocessed_dataset()
    dataset_tokenized: Dataset = cached_dataset or imdb_dataset.map(  # type: ignore
        # batched
        lambda examples: {
            "target_reward": sentiment_reward.reward_batch(
                examples["text"], batch_size=32
            )
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
    # log training target_reward
    training_reward_dist = calculate_distribution_statistics(
        dist=dataset_tokenized["train"]["target_reward"]  # type: ignore
    )
    print(f"Training target_reward distribution: {training_reward_dist}")
    # save the preprocessed dataset if we didn't already have it
    if not cached_dataset:
        dataset_tokenized.save_to_disk(preprocessed_dataset_path)
    dataset_tokenized.set_format(
        type="torch",
        columns=[
            "input_ids",
            "target_reward",
            "attention_mask",
        ],
    )

    print("ok")

    # Train the model using the device
    gpt2_model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model.value).to(
        device
    )
    loaded_model = ModifiedGPT2LMHeadModel(existing_head_model=gpt2_model)

    training_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=loaded_model,
        args=training_args,
        train_dataset=dataset_tokenized["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    # Save the model
    loaded_model.save_pretrained(save_dir)
    test_text: List[str] = dataset_tokenized["test"]["text"]  # type: ignore [call-overload]
    evaluate_test_set(
        test_text=test_text,
        model=loaded_model,
        tokenizer=tokenizer,
        sentiment_reward=sentiment_reward,
        limit=1000,
    )


if __name__ == "__main__":
    # run with
    # export PYTHONPATH=.; python examples/imdb/train_imdb.py --batch-size 10 --epochs 1
    # export PYTHONPATH=.; python examples/imdb/train_imdb.py --batch-size 1 --epochs 1 --model gpt2-xl
    typer.run(main)
