from typing import List

import torch
from transformers import Pipeline, pipeline


class ImdbRewardModel:
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
