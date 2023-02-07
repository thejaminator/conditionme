from datasets import Dataset, DatasetDict
from transformers import GPT2LMHeadModel, AutoTokenizer

from conditionme.modified_gpt2_lm_head import ModifiedGPT2LMHeadModel
from examples.imdb.train_imdb import train_imdb
from tests.sanity.test_evaluate_test_set import MockImdbRewardModel


def test_train():
    # Contains train, test set.
    dataset = {
        "train": Dataset.from_dict(
            {"text": ["this is a test", "this is another test"]}
        ),
        "test": Dataset.from_dict({"text": ["this is a test", "this is another test"]}),
    }
    dataset_hg = DatasetDict(dataset)
    tiny_model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    conditional_model: ModifiedGPT2LMHeadModel = ModifiedGPT2LMHeadModel(
        existing_head_model=tiny_model
    )
    tiny_tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tiny_tokenizer.pad_token = tiny_tokenizer.eos_token
    sentiment_reward_model = MockImdbRewardModel(device="cpu")
    train_imdb(
        batch_size=1,
        epochs=1,
        save_dir="saved",
        tokenizer=tiny_tokenizer,
        gpt2_model=conditional_model,
        reward_model=sentiment_reward_model,
        learning_rate=0.0001,
        dataset=dataset_hg,
    )
