from abc import ABC, abstractmethod

import torch


class RewardHandler(ABC):
    @abstractmethod
    def handle_reward(
        self, target_reward: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        pass


class DefaultRewardHandler(RewardHandler):
    def handle_reward(
        self, target_reward: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # the first token (assume to be eos) will have the target_reward added to it
        # This is added to the last dimension of hidden_states
        modified_hidden_states = hidden_states.clone()
        modified_hidden_states[:, 0, -1] += target_reward
        return modified_hidden_states
