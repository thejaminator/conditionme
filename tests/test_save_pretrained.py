from transformers import GPT2LMHeadModel

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel


def test_save_pretrained():
    tiny_model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    conditional_model: ModifiedGPT2LMHeadModel = ModifiedGPT2LMHeadModel(
        existing_head_model=tiny_model
    )
    state_dict = conditional_model.state_dict()
    conditional_model.save_pretrained("tests/saved")
    print('yes')