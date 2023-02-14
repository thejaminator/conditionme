## Introduction
`conditionme` is library for easily retraining existing language models to work in a decision tranformer / upside down rl fashion.

We eventually hope it can be something similar to [trl](https://github.com/lvwerra/trl), just that instead of PPO we'll train in a decision transformer fashion.
This still a very early stage library, so expect bugs and missing features.



## Why does this library exist?
I haven't found a library that allows you to easily retrain existing language models (e.g. gpt2, gpt-j) to work in a decision tranformer / upside down rl fashion.
There could be some aspects for training in a decision transformer fashion that could

## How does it work?
We can't take the [decision transformer implementation](https://huggingface.co/blog/decision-transformers) and just make our existing language model work with the decision transformer architecture. 
There's a few simplifications that we need to make it work.
1. Instead of multiple (reward-to-go, state, action) in an rollout/episode, we only have one single reward per episode. 
2. Rather than having separate state and action heads, we'll continue using the same language model head. 
So it becomes (reward-to-go, text completion) instead.

What we do is:
1. We reserve the first token to encode the target reward.
2. The target reward is added all values of the hidden state of the first token. 
The positional encoding still remains in the hidden state, so that the model can learn to condition on the reward token to affect the output.
3. We finetune our model autoregressively, just that we'll specify the target reward along with our inputs.

## Toy example - Imdb sentiment analysis
Using gpt-large as our pretrained model, we finetune our model to match our target reward.
View the training script [here](examples/imdb/train_imdb.py).

```bash
git clone git+https://github.com/thejaminator/conditionme.git#egg=conditionme
cd conditionme
# make a virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.; python examples/imdb/train_imdb.py --batch-size 1 --epochs 1 --model gpt2-large --save-dir gpt2_conditional
```

| ![high_reward_dist.png](eval_results%2Flarge_results%2Fhigh_reward_dist.png) | ![low_reward_dist.png](eval_results%2Flarge_results%2Flow_reward_dist.png) |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Actual reward obtained by setting the target reward to 1.0                    | Actual reward obtained by setting the target reward to 0.0                 |

We observe that we can obtain either very bad or very good movie reviews, controlled by the target reward we set.

See csv of results [here](eval_results/large_results)

Note: if you try to plot a correlation plot between the target reward and the actual reward, it may look like it doesn't work well between the range of target reward (0.1, 0.9) . This is probably because the training dataset is heavily skewed towards 0.0 or 1.0 rewards.
<details>
  <summary>See correlation plot</summary>

![correlation.png](eval_results%2Flarge_results%2Fcorrelation.png)
</details>



## How it works - details

We reserve the first token to encode the target reward. Or rather, the first unmasked token. 
This is because we pad to the left for batched training, so we can't just use the first token - it would be a pad token that is masked out.

This means that when we pass input ids to the model for training, the first token id will always be reserved to be the target reward token.
This library *should* handle the details of this happening. You'll just need to specify what the reserved token should be.

The token should be not an eos token, because in some huggingface data collators they mask out eos token ids. That causes your reward token to be masked out.

NOTE: Another way of doing this is to literally encode the reward as text input. A downside of this is that you'll probably be more open to prompt injection. [I demonstrate it here](https://github.com/thejaminator/prompt_reward_rl/blob/main/documentation/main_page.md#ability-to-match-a-single-reward)
And you'll need to be more careful with how your rewards can get tokenized into multiple different tokens.



## TODO list
- [x] Validate that it works on a toy example
- [ ] Reach out to others and ask if the hack makes sense
- [ ] Add support for huggingface pretrained models saving
- [ ] Add examples for RLHF tasks - e.g. Openai's summarization where an [existing reward model is already available](https://huggingface.co/OpenAssistant)
- [ ] Add support for some other pretrained models - not just gpt2
- [ ] Write docs on how to make it work with other pretrained models that are not added yet.
- [ ] Add support for online training
