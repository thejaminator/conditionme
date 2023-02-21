import torch
from datasets import load_dataset, Dataset

from conditionme.reward_models.imdb_reward_model import ImdbRewardModel

if __name__ == "__main__":
    gpu_device = torch.device("cuda:0")
    sentiment_reward = ImdbRewardModel(device=gpu_device)
    dataset_rewarded: Dataset = load_dataset("imdb").map(
        # batched
        lambda examples: {"target_rewards": sentiment_reward.reward_batch(examples["text"], batch_size=32)},
        batched=True,
    )
    # export to json
    dataset_rewarded.to_json("imdb_dataset_rewarded.json")
    # To run:
    # export PYTHONPATH=.; python examples/imdb/export_imdb_reward_dataset.py
