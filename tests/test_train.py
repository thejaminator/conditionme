from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import GPT2LMHeadModel, AutoTokenizer

from conditionme.decision_gpt2_tokenize import create_decision_tokenizer
from conditionme.decision_gpt2_lm_head import DecisionGPT2LMHeadModel
from conditionme.scaling.scaler import Times1000
from examples.imdb.train_imdb import train_imdb
from tests.test_evaluate_test_set import MockImdbRewardModel


def test_train(tmp_path: Path):
    # Contains train, test set.
    dataset = {
        "train": Dataset.from_dict(
            {"text": ["this", "this is another test", "third test" * 5000]}
        ),
        "test": Dataset.from_dict({"text": ["this is a test", "this is another test"]}),
    }
    dataset_hg = DatasetDict(dataset)
    tiny_model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    conditional_model: DecisionGPT2LMHeadModel = (
        DecisionGPT2LMHeadModel.from_loaded_pretrained_model(loaded_model=tiny_model)
    )
    tiny_tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    decision_tokenizer = create_decision_tokenizer(tiny_tokenizer)
    sentiment_reward_model = MockImdbRewardModel(device="cpu")
    train_imdb(
        batch_size=4,
        epochs=1,
        save_dir=tmp_path,
        decision_tokenizer=decision_tokenizer,
        decision_model=conditional_model,
        reward_model=sentiment_reward_model,
        learning_rate=0.0001,
        dataset=dataset_hg,
        scaler_type=Times1000,
    )
