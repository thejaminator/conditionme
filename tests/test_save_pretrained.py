from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from conditionme.decision_gpt2_lm_head import DecisionGPT2LMHeadModel


def test_save_pretrained(tmp_path: Path):
    tiny_model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    conditional_model: DecisionGPT2LMHeadModel = (
        DecisionGPT2LMHeadModel.from_loaded_pretrained_model(loaded_model=tiny_model)
    )
    state_dict = conditional_model.state_dict()
    conditional_model.save_pretrained(tmp_path)
    loaded_model = DecisionGPT2LMHeadModel.from_pretrained(tmp_path)
    loaded_state_dict = loaded_model.state_dict()
    # assert that allclose
    for key in state_dict:
        # Check if the weights are the same
        torch.allclose(state_dict[key], loaded_state_dict[key])
