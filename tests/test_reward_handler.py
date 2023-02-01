import torch

from conditionme.reward_handler import DefaultRewardHandler


def test_default():
    handler = DefaultRewardHandler()
    # E.g. 3 examples in a batch
    target_reward = torch.tensor([1.0, 10.0, 100.0])
    # 3 x 2 x 2 hidden states
    hidden_states = torch.tensor(
        [
            # first sequence
            [
                [0.0, 1.0],  # first token
                [-1, -1],
            ],
            # second sequence
            [
                [0.0, 1.0],  # first token
                [-1, -1],
            ],
            # third sequence
            [
                [0.0, 1.0],  # first token
                [-1, -1],
            ],
        ]
    )
    # The reward token here are all the first tokens
    target_reward_position = torch.tensor([0, 0, 0])
    modified_hidden_states = handler.handle_reward(
        target_reward, hidden_states, target_reward_position=target_reward_position,
        past_length=0,
    )
    assert modified_hidden_states[0, 0, -1] == 2.0, "should add 1.0 to the first token"
    assert (
        modified_hidden_states[1, 0, -1] == 11.0
    ), "should add 10.0 to the first token"
    assert (
        modified_hidden_states[2, 0, -1] == 101.0
    ), "should add 100.0 to the first token"


def test_default_different_reward_positions():
    handler = DefaultRewardHandler()
    # E.g. 3 examples in a batch
    target_reward = torch.tensor([1.0, 1.0, 1.0])
    # 3 x 2 x 2 hidden states
    hidden_states = torch.tensor(
        [
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ]
    )
    # Masking applied.
    target_reward_position = torch.tensor(
        [
            1,  # First sequence has reward at second token
            0,  # Second sequence has reward at first token
            1,  # Third sequence has reward at second token
        ]
    )
    modified_hidden_states = handler.handle_reward(
        target_reward, hidden_states, target_reward_position=target_reward_position,
        past_length=0,
    )
    # first sequence
    assert (
        modified_hidden_states[0, 0, -1] == 0.0
    ), "should not add 1.0 to the first token"
    assert modified_hidden_states[0, 1, -1] == 1.0, "should add 1.0 to the second token"
    # second sequence
    assert modified_hidden_states[1, 0, -1] == 1.0, "should add 1.0 to the first token"
    # third sequence
    assert (
        modified_hidden_states[2, 0, -1] == 0.0
    ), "should not add 1.0 to the first token"
    assert modified_hidden_states[2, 1, -1] == 1.0, "should add 1.0 to the second token"
