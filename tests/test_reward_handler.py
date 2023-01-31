import torch

from conditionme.reward_handler import DefaultRewardHandler


def test_default():
    handler = DefaultRewardHandler()
    # E.g. 3 examples in a batch
    target_reward = torch.tensor([1.0, 10.0, 100.0])
    # 3 x 2 x 2 hidden states
    hidden_states = torch.tensor(
        [
            [
                [0.0, 1.0],
                [-1, -1],
            ],
            [
                [0.0, 1.0],
                [-1, -1],
            ],
            [
                [0.0, 1.0],
                [-1, -1],
            ],
        ]
    )
    modified_hidden_states = handler.handle_reward(target_reward, hidden_states)
    assert modified_hidden_states[0, 0, -1] == 2.0, "should add 1.0 to the first token"
    assert (
        modified_hidden_states[1, 0, -1] == 11.0
    ), "should add 10.0 to the first token"
    assert (
        modified_hidden_states[2, 0, -1] == 101.0
    ), "should add 100.0 to the first token"
