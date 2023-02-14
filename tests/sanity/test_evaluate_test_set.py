import random
from pathlib import Path
from typing import List

from transformers import GPT2LMHeadModel, AutoTokenizer

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel
from conditionme.normalization.normalizer import DoNothingNormalizer
from examples.imdb.evaluate_imdb import evaluate_test_set
from examples.imdb.imdb_reward_model import ImdbRewardModel


class MockImdbRewardModel(ImdbRewardModel):
    def __init__(self, device):
        pass

    def reward_single(self, text: str) -> float:
        # return 0 to 1 float random
        # use text as a seed
        randomizer = random.Random(text)
        return randomizer.random()

    def reward_batch(self, texts: List[str], batch_size: int = 16) -> List[float]:
        return [self.reward_single(text) for text in texts]


def test_evaluate_test_set():
    test_text = ["this", "this is another test test test"]
    tiny_model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    conditional_model: ModifiedGPT2LMHeadModel = (
        ModifiedGPT2LMHeadModel.from_loaded_pretrained_model(loaded_model=tiny_model)
    )
    tiny_tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    sentiment_reward_model = MockImdbRewardModel(device="cpu")
    normalizer = DoNothingNormalizer()
    save_dir = Path("tests/")
    evaluate_test_set(
        test_text=test_text,
        model=conditional_model,
        tokenizer=tiny_tokenizer,
        sentiment_reward=sentiment_reward_model,
        normalizer=normalizer,
        limit=10,
        save_dir=save_dir,
    )
