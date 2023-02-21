from typing import List

from datasets import Dataset, DatasetDict

from conditionme.normalization.normalizer import Times1000
from conditionme.reward_models.imdb_reward_model import ImdbRewardModel
from examples.imdb.train_imdb import batch_normalize


def test_batch_normalize():
    dataset = {
        "train": Dataset.from_dict(
            {"text": ["this", "this is another test", "third test" * 5000]}
        ),
        "test": Dataset.from_dict({"text": ["this is a test", "this is another test"]}),
    }
    dataset_hg = DatasetDict(dataset)

    class MockImdbRewardModel(ImdbRewardModel):
        def __init__(self, device):
            pass

        def reward_single(self, text: str) -> float:
            return 1

        def reward_batch(self, texts: List[str], batch_size: int = 16) -> List[float]:
            return [self.reward_single(text) for text in texts]

    reward_model = MockImdbRewardModel(device="cpu")

    dataset_rewarded: Dataset = dataset_hg.map(  # type: ignore
        # batched
        lambda examples: {
            "target_rewards": reward_model.reward_batch(examples["text"], batch_size=32)
        },
        batched=True,
    )
    normalizer = Times1000()
    dataset_normalized = dataset_rewarded.map(
        lambda x: batch_normalize(x, normalizer=normalizer), batched=True
    )
    assert dataset_normalized["train"]["target_rewards"][0] == 1000  # type: ignore
