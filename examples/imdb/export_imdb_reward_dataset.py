import torch
from datasets import load_dataset, Dataset, DatasetDict

from conditionme.reward_models.imdb_reward_model import ImdbRewardModel

if __name__ == "__main__":
    gpu_device = torch.device("cuda:0")
    sentiment_reward = ImdbRewardModel(device=gpu_device)
    dataset_rewarded_dict: DatasetDict = load_dataset("imdb").map(
        # batched
        lambda examples: {"target_rewards": sentiment_reward.reward_batch(examples["text"], batch_size=32)},
        batched=True,
    )
    dataset_rewarded_train: Dataset = dataset_rewarded_dict["train"]
    dataset_rewarded_test: Dataset = dataset_rewarded_dict["test"]
    # export to json
    dataset_rewarded_train.to_json("imdb_train.json")
    dataset_rewarded_test.to_json("imdb_test.json")
    # To run:
    # export PYTHONPATH=.; python examples/imdb/export_imdb_reward_dataset.py
