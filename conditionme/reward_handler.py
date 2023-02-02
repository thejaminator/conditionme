from abc import ABC, abstractmethod

import torch


class RewardHandler(ABC):
    @abstractmethod
    def handle_reward(
        self,
        target_reward: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,  # 2D tensor of shape (batch_size, sequence_length)
        past_length: int,
    ) -> torch.Tensor:
        pass


class DefaultRewardHandler(RewardHandler):
    def __init__(self, reward_token_id: int):
        self.reward_token_id = reward_token_id

    def handle_reward(
        self,
        target_reward: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,  # 2D tensor of shape (batch_size, sequence_length)
        past_length: int,
    ) -> torch.Tensor:
        # the first token that matches reward_token_id will have the target_reward added to it
        # This is added to the last dimension of hidden_states
        modified_hidden_states = hidden_states.clone()
        # Add the target reward to the first token that matches reward_token_id
        if (
            past_length != 0
        ):  # This is a hack to know when we are decoding tokens one by one
            return modified_hidden_states
        # Get the index of the first token that matches reward_token_id
        indexes = find_reward_token_position(input_ids, self.reward_token_id)
        for row, index in enumerate(indexes):
            modified_hidden_states[row, index, :] += target_reward[row]

        return modified_hidden_states


def find_reward_token_position(
    input_ids: torch.LongTensor,  # 2D tensor of shape (batch_size, sequence_length)
    reward_token_id: int,
) -> torch.LongTensor:  # 1D tensor of length batch_size
    # Get the indices of the reward token in each sequence
    indexes: torch.LongTensor = torch.zeros(input_ids.shape[0], dtype=torch.long)  # type: ignore
    for row, inputs in enumerate(input_ids):
        item = (inputs == reward_token_id).nonzero()
        assert (
            item.nelement() != 0
        ), f"Could not find the reward token {reward_token_id} in the input_ids {inputs.tolist()}"
        index = item[0]
        indexes[row] = index

    # Return the first occurrence of the reward token in each sequence
    return indexes


class AddRewardToWholeEosHandler(RewardHandler):
    def __init__(self, reward_token_id: int):
        self.reward_token_id = reward_token_id

    def handle_reward(
        self,
        target_reward: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        past_length: int,
    ) -> torch.Tensor:
        ## the first token that matches reward_token_id will have the target_reward added to it
        # This is added to the last dimension of hidden_states
        modified_hidden_states = hidden_states.clone()
        # Add the target reward to the first token that matches reward_token_id
        if (
            past_length != 0
        ):  # This is a hack to know when we are decoding tokens one by one
            return modified_hidden_states
        # Get the index of the first token that matches reward_token_id
        indexes = find_reward_token_position(input_ids, self.reward_token_id)
        for row, index in enumerate(indexes):
            modified_hidden_states[row, index, :] += target_reward[row]

        return modified_hidden_states


class ReplaceRewardToWholeEosHandler(RewardHandler):
    def __init__(self, reward_token_id: int):
        self.reward_token_id = reward_token_id

    def handle_reward(
        self,
        target_reward: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        past_length: int,
    ) -> torch.Tensor:
        ## the first token that matches reward_token_id will have the target_reward added to it
        # This is added to the last dimension of hidden_states
        modified_hidden_states = hidden_states.clone()
        # Add the target reward to the first token that matches reward_token_id
        if (
            past_length != 0
        ):  # This is a hack to know when we are decoding tokens one by one
            return modified_hidden_states
        # Get the index of the first token that matches reward_token_id
        indexes = find_reward_token_position(input_ids, self.reward_token_id)
        for row, index in enumerate(indexes):
            modified_hidden_states[row, index, :] = target_reward[row]

        return modified_hidden_states
