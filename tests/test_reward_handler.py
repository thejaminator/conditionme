import torch

from conditionme.reward_handler import DefaultRewardHandler, find_reward_token_position


def test_find_reward_token_position():
    input_ids = torch.LongTensor(
        [
            [5, 12312],
            [5, 2313],
            [5, 2],
        ],
    )
    reward_token_id = 5
    assert (
        find_reward_token_position(input_ids, reward_token_id).tolist()
        == torch.LongTensor([0, 0, 0]).tolist()
    )

    input_ids = torch.LongTensor([[5, 12312], [5, 2313], [2, 5]])
    reward_token_id = 5
    assert (
        find_reward_token_position(input_ids, reward_token_id).tolist()
        == torch.LongTensor([0, 0, 1]).tolist()
    )


def test_default():
    handler = DefaultRewardHandler(reward_token_id=5)
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
    reward_token_id = 5
    input_ids = torch.LongTensor(
        [
            [5, 12312],
            [5, 2313],
            [5, 2],
        ],
    )

    modified_hidden_states = handler.handle_reward(
        target_reward,
        hidden_states,
        past_length=0,
        input_ids=input_ids,
    )
    assert modified_hidden_states[0, 0, -1] == 2.0, "should add 1.0 to the first token"
    assert (
        modified_hidden_states[1, 0, -1] == 11.0
    ), "should add 10.0 to the first token"
    assert (
        modified_hidden_states[2, 0, -1] == 101.0
    ), "should add 100.0 to the first token"


def test_default_different_reward_positions():
    handler = DefaultRewardHandler(reward_token_id=5)
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
    input_ids = torch.LongTensor(
        [
            [1, 5],  # First sequence has reward at second token,
            [5, 1],  # Second sequence has reward at first token
            [1, 5],  # Third sequence has reward at second token
        ],
    )
    modified_hidden_states = handler.handle_reward(
        target_reward=target_reward,
        hidden_states=hidden_states,
        input_ids=input_ids,
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
