from abc import ABC, abstractmethod

import torch


class RewardHandler(ABC):
    @abstractmethod
    def handle_reward(
        self,
        target_reward: torch.Tensor,
        hidden_states: torch.Tensor,
        target_reward_position: torch.Tensor,
        past_length: int,
    ) -> torch.Tensor:
        pass


class DefaultRewardHandler(RewardHandler):
    def handle_reward(
        self,
        target_reward: torch.Tensor,
        hidden_states: torch.Tensor,
        target_reward_position: torch.Tensor,  # 1D tensor of length batch_size
        past_length: int
    ) -> torch.Tensor:
        # the first token (assume to be eos) will have the target_reward added to it
        # This is added to the last dimension of hidden_states
        modified_hidden_states = hidden_states.clone()
        # Add the target reward to the first token
        for sequence_dim, position in enumerate(target_reward_position):
            # Big hack. TODO: detect if we are decoding one token by one token in a better
            if past_length == 0:
                modified_hidden_states[sequence_dim, position, -1] += target_reward[
                    sequence_dim
                ]
        fn = modified_hidden_states.grad_fn
        return modified_hidden_states
