import torch

from conditionme.reward_handler import DefaultRewardHandler


def test_default():
    handler = DefaultRewardHandler()
    target_reward = torch.tensor([1.0])
    hidden_states = torch.tensor([[1.0, 2.0, 3.0]])
    modified_hidden_states = handler.handle_reward(target_reward, hidden_states)
    assert modified_hidden_states[0, 0, -1] == 2.0
