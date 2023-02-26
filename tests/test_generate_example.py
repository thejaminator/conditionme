def test_model_generate():
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from conditionme import create_decision_tokenizer, DecisionGPT2LMHeadModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    decision_tokenizer = create_decision_tokenizer(tokenizer)
    loaded_model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    decision_model = DecisionGPT2LMHeadModel.from_loaded_pretrained_model(loaded_model)
    encoded_text = decision_tokenizer.encode("this is a test")
    generated = decision_model.generate(
        input_ids=torch.tensor([encoded_text]),
        target_rewards=torch.tensor([1.0]),
    )
    generated_text = decision_tokenizer.decode(generated[0])
