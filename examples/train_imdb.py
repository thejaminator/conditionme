"""
Trains GPT2 on IMDB dataset
"""
from typing import List

import torch
from datasets import load_dataset
from datasets.formatting.formatting import LazyBatch
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    Pipeline,
    pipeline,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    GPT2LMHeadModel,
)

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
eos_token: str = tokenizer.eos_token
# see https://github.com/huggingface/transformers/issues/2630
tokenizer.pad_token = tokenizer.eos_token


class Rewarder:
    # An example of a possible reward function using sentiment analysis
    def __init__(self, device: torch.device):
        self.sentiment_pipe: Pipeline = pipeline(
            "sentiment-analysis",
            model="lvwerra/distilbert-imdb",
            device=device,
            truncation_strategy="longest_first",
            truncation=True,
            max_length=512,
        )

    def reward_single(self, text: str) -> float:
        """
        Computes the reward for atext
        """

        pipe_outputs = self.sentiment_pipe(text, return_all_scores=True)
        rewards = [output[1]["score"] for output in pipe_outputs]
        assert len(rewards) == 1
        return rewards[0]

    def reward_batch(self, texts: List[str], batch_size: int = 16) -> List[float]:
        """
        Computes the reward for a batch of texts
        """
        # you need to truncate the text to a maximum token length
        # truncated = [text[:512] for text in texts]
        pipe_outputs = self.sentiment_pipe(
            texts, return_all_scores=True, batch_size=batch_size
        )
        rewards = [output[1]["score"] for output in pipe_outputs]
        return rewards


def tokenize(examples: LazyBatch) -> BatchEncoding:
    # add the eos token to the start of all text
    # the `forward` method of ModifiedGPT2LMHeadModel will modify the embedding of the eos token
    # to include the reward
    new_text = [eos_token + text for text in examples["text"]]
    tokenizer_result = tokenizer(new_text, truncation=True)
    # add the precomputed reward to the result
    tokenizer_result["target_reward"] = examples["target_reward"]
    tokenizer_result["labels"] = tokenizer_result["input_ids"].copy()
    return tokenizer_result


def main():
    # Download and tokenize the dataset
    imdb_dataset = load_dataset("imdb")
    # limit the dataset to 100 examples
    limit = 100
    imdb_dataset_limited = imdb_dataset["train"].select(range(limit))
    # compute the reward for each example
    # prefer gpu if available
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    sentiment_reward = Rewarder(device=device)
    dataset_tokenized = imdb_dataset_limited.map(
        # batched
        lambda examples: {
            "target_reward": sentiment_reward.reward_batch(examples["text"])
        },
        batched=True,
        batch_size=16,
    ).map(
        tokenize,
        batched=True,
    )
    dataset_tokenized.set_format(type="torch", columns=["input_ids", "target_reward", "labels"])
    print("ok")

    # Train the model using the device
    gpt2_model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained("gpt2").to(
        device
    )
    model = ModifiedGPT2LMHeadModel(existing_head_model=gpt2_model)
    training_args = TrainingArguments(
        output_dir="./gpt2-imdb",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
