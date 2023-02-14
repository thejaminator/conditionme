import torch
from transformers import GPT2LMHeadModel

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel


def test_save_pretrained():
    tiny_model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    conditional_model: ModifiedGPT2LMHeadModel = (
        ModifiedGPT2LMHeadModel.from_loaded_pretrained_model(loaded_model=tiny_model)
    )
    state_dict = conditional_model.state_dict()
    conditional_model.save_pretrained("tests/saved")
    loaded_model = ModifiedGPT2LMHeadModel.from_pretrained("tests/saved")
    loaded_state_dict = loaded_model.state_dict()
    # assert that allclose
    for key in state_dict:
        # Check if the weights are the same
        torch.allclose(state_dict[key], loaded_state_dict[key])
