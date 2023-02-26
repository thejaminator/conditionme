import dataclasses
from typing import Sequence, Dict

import pandas as pd
from slist import Slist

from conditionme.completion.complete_model import PromptCompletion


@dataclasses.dataclass
class RewardEvaluationRow:
    prompt: str
    completion: str
    target_reward: float
    actual_reward: float


def reward_evaluation_rows(
    prompt_completions: Sequence[PromptCompletion],
    target_rewards: Sequence[float],
    actual_rewards: Sequence[float],
) -> Slist[RewardEvaluationRow]:
    return (
        Slist(prompt_completions)
        .zip(target_rewards, actual_rewards)
        .map(
            lambda p: RewardEvaluationRow(
                prompt=p[0].prompt,
                completion=p[0].completion,
                target_reward=p[1],
                actual_reward=p[2],
            )
        )
    )


# create a pandas dataframe from the reward evaluation rows
def reward_evaluation_table(
    evaluation_rows: Sequence[RewardEvaluationRow],
) -> pd.DataFrame:
    dicts: Slist[Dict] = Slist(evaluation_rows).map(dataclasses.asdict)
    return pd.DataFrame(dicts)
