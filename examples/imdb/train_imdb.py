"""
Trains GPT2 on IMDB dataset
"""
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Type

import torch
import typer
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BatchEncoding,
    GPT2LMHeadModel,
)

from conditionme.decision_gpt2_lm_head import DecisionGPT2LMHeadModel
from conditionme.decision_gpt2_tokenize import batch_tokenize_gpt2, create_decision_tokenizer, DecisionTokenizer
from conditionme.reward_models.imdb_reward_model import ImdbRewardModel
from conditionme.scaling.scaler import (
    RewardScaler,
    get_scaler, ScalerOptions,
)
from conditionme.statistics.calculate_distribution import (
    calculate_distribution_statistics,
)
from examples.imdb.evaluate_imdb import evaluate_test_set


class GPT2ModelOptions(Enum):
    # see https://huggingface.co/transformers/v2.2.0/pretrained_models.html
    gpt2 = "gpt2"
    gpt2_medium = "gpt2-medium"
    gpt2_large = "gpt2-large"
    gpt2_xl = "gpt2-xl"


def batch_scale(
    batch: BatchEncoding,
    scaler: RewardScaler,
) -> BatchEncoding:
    # scale the reward
    scaled_reward = scaler.scale_rewards(batch["target_rewards"])
    # replace the reward with the scaled reward
    batch["target_rewards"] = scaled_reward
    return batch


def train_imdb(
    batch_size: int,
    epochs: int,
    save_dir: Path,
    decision_tokenizer: DecisionTokenizer,
    decision_model: DecisionGPT2LMHeadModel,
    learning_rate: float,
    # must contain "train", "test", and "text" keys
    dataset: Union[DatasetDict, Dataset],
    reward_model: ImdbRewardModel,
    scaler_type: Type[RewardScaler],
) -> None:

    dataset_tokenized: Dataset = dataset.map(  # type: ignore
        # batched
        lambda examples: {"target_rewards": reward_model.reward_batch(examples["text"], batch_size=32)},
        batched=True,
    ).map(
        lambda x: batch_tokenize_gpt2(
            x["text"],
            target_rewards=x["target_rewards"],
            decision_tokenizer=decision_tokenizer,
            add_eos_at_end=True,
        ),
        batch_size=batch_size,  # We don't have to pad so much if batch_size is smaller
        batched=True,
    )
    scaler: RewardScaler = scaler_type.from_rewards(
        rewards=dataset_tokenized["train"]["target_rewards"]  # type: ignore
    )
    # update the dataset with the scaled rewards
    scaled_dataset = dataset_tokenized.map(lambda x: batch_scale(x, scaler=scaler), batched=True)

    # Save the scaler
    scaler.save_scaler(Path(save_dir))

    # log training target_rewards
    training_reward_dist = calculate_distribution_statistics(
        dist=scaled_dataset["train"]["target_rewards"]  # type: ignore
    )
    print(f"Training target_rewards distribution: {training_reward_dist}")
    scaled_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "target_rewards",
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
    trainer = Trainer(
        model=decision_model,
        args=training_args,
        train_dataset=scaled_dataset["train"],
        tokenizer=decision_tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=decision_tokenizer, mlm=False),
    )
    trainer.train()

    # Save the model
    decision_model.save_pretrained(save_dir)
    test_text: List[str] = scaled_dataset["test"]["text"]  # type: ignore [call-overload]
    evaluate_test_set(
        test_text=test_text,
        model=decision_model,
        decision_tokenizer=decision_tokenizer,
        sentiment_reward=reward_model,
        limit=1000,
        scaler=scaler,
        save_dir=Path(save_dir),
    )


def main(
    batch_size: int = 1,
    epochs: int = 1,
    save_dir: str = "gdrive/My Drive/conditionme",
    model: GPT2ModelOptions = GPT2ModelOptions.gpt2,
    learning_rate: float = 5e-5,
    device: Optional[str] = None,
    scaler: ScalerOptions = ScalerOptions.do_nothing,
):

    scaler_type: Type[RewardScaler] = get_scaler(scaler)
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
        else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    )
    sentiment_reward = ImdbRewardModel(device=device_selected)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    decision_tokenizer = create_decision_tokenizer(tokenizer)
    loaded_model = GPT2LMHeadModel.from_pretrained(model.value)
    gpt2_model = DecisionGPT2LMHeadModel.from_loaded_pretrained_model(loaded_model).to(device)
    train_imdb(
        batch_size=batch_size,
        epochs=epochs,
        save_dir=Path(save_dir),
        decision_tokenizer=decision_tokenizer,
        decision_model=gpt2_model,
        learning_rate=learning_rate,
        dataset=imdb_dataset,
        reward_model=sentiment_reward,
        scaler_type=scaler_type,
    )


if __name__ == "__main__":
    # run with
    # export PYTHONPATH=.; python examples/imdb/train_imdb.py --batch-size 10 --epochs 1
    # export PYTHONPATH=.; python examples/imdb/train_imdb.py --batch-size 1 --epochs 1 --model gpt2-medium --save-dir saved_medium --scaler min_max
    typer.run(main)
