from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from conditionme.decision_gpt2_lm_head import DecisionGPT2LMHeadModel
from conditionme.completion.complete_model import (
    complete_text_with_reward_batched,
    PromptCompletion,
)
from conditionme.decision_gpt2_tokenize import batch_tokenize_gpt2, create_decision_tokenizer


def test_gpt_sanity(tmp_path: Path):
    # create fake dataset for huggingface dataset
    dataset = {
        "text": [
            "This is a test",
            "This is another test",
            "This is a third test" * 1000,
        ],
        "target_rewards": [0.1, 0.2, 0.3],
    }
    huggingface_dataset: Dataset = Dataset.from_dict(dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        "sshleifer/tiny-gpt2", padding_side="left"
    )
    decision_tokenizer = create_decision_tokenizer(tokenizer)


    # tokenize the dataset
    dataset_tokenized = huggingface_dataset.map(
        lambda examples: batch_tokenize_gpt2(
            text=examples["text"],
            target_rewards=examples["target_rewards"],
            decision_tokenizer=decision_tokenizer,
            add_eos_at_end=True,
        ),
        batched=True,
        batch_size=3,
    )
    dataset_tokenized.set_format(
        type="torch",
        columns=[
            "input_ids",
            "target_rewards",
            "attention_mask",
        ],
    )
    device: torch.device = torch.device("cpu")
    # Train the model using the device
    gpt2_model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(
        "sshleifer/tiny-gpt2"
    ).to(device)
    model = DecisionGPT2LMHeadModel.from_loaded_pretrained_model(
        loaded_model=gpt2_model
    )

    # Optionally save to drive
    training_args = TrainingArguments(
        output_dir=tmp_path,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        save_steps=1,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=decision_tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(tmp_path)
    new_model = DecisionGPT2LMHeadModel.from_pretrained(tmp_path)


def test_complete_text_with_reward_batched():
    # create fake dataset for huggingface dataset
    dataset = {
        "text": [
            "This is a test",
            "This is another test",
            "This is thirdthirdthirdthird test test test",
        ],
        "target_rewards": [0.1, 0.2, 0.3],
    }
    device: torch.device = torch.device("cpu")
    # Train the model using the device
    gpt2_model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(
        "sshleifer/tiny-gpt2"
    ).to(device)
    model = DecisionGPT2LMHeadModel.from_loaded_pretrained_model(
        loaded_model=gpt2_model
    )
    huggingface_dataset: Dataset = Dataset.from_dict(dataset)
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    decision_tokenizer = create_decision_tokenizer(tokenizer)
    # Take 500 test set
    # convert into a list of space separated tokens
    test_text_tokenized: List[List[str]] = [
        text.split(" ") for text in huggingface_dataset["text"]
    ]
    # take the first 3 tokens from each list
    first_3_tokens_list: List[List[str]] = [text[:3] for text in test_text_tokenized]
    # join the first 3 tokens into a string
    first_3_tokens: List[str] = [" ".join(text) for text in first_3_tokens_list]
    # complete the text
    completions: List[PromptCompletion] = complete_text_with_reward_batched(
        prompts=first_3_tokens,
        model=model,
        tokenizer=decision_tokenizer,
        target_rewards=[1.0] * len(first_3_tokens),
        max_new_tokens=1,
    )
    assert len(completions) == len(test_text_tokenized)
