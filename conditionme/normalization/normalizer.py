import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Sequence, Dict, List, Type

from slist import Slist

from conditionme.type_check.utils import assert_not_none, assert_never


class RewardNormalizer(ABC):
    @staticmethod
    @abstractmethod
    def from_rewards(rewards: Sequence[float]) -> "RewardNormalizer":
        raise NotImplementedError

    @abstractmethod
    def normalize_reward(self, reward: float) -> float:
        raise NotImplementedError

    def normalize_rewards(self, rewards: Sequence[float]) -> List[float]:
        return [self.normalize_reward(reward) for reward in rewards]

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def save_normalizer(self, path: Path) -> None:
        # Create the directory if it does not exist
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "normalizer"
        with open(file_path, "w") as f:
            print(f"Saving normalizer to {file_path}")
            f.write(self.to_json())

    @classmethod
    def load_normalizer(cls, path: Path) -> "RewardNormalizer":
        file_path = path / "normalizer"
        with open(file_path, "r") as f:
            print(f"Loading normalizer from {file_path}")
            _dict = json.load(f)
        return cls.create_from_dict(_dict)

    @staticmethod
    def create_from_dict(_dict: Dict[str, Any]) -> "RewardNormalizer":
        name = _dict["name"]
        if name == MinMaxNormalizer.name():
            return MinMaxNormalizer.from_dict(_dict)
        elif name == StandardScaleNormalizer.name():
            return StandardScaleNormalizer.from_dict(_dict)
        elif name == DoNothingNormalizer.name():
            return DoNothingNormalizer()
        elif name == Times1000.name():
            return Times1000()
        elif name == StandardTimes1000Normalizer.name():
            return StandardTimes1000Normalizer.from_dict(_dict)
        else:
            raise ValueError(f"Unknown normalizer name: {name}")


class MinMaxNormalizer(RewardNormalizer):
    # A normalizer that will normalize the rewards to be between 0 and 1
    def __init__(
        self,
        reward_min: float,
        reward_max: float,
    ):
        self.reward_min = reward_min
        self.reward_max = reward_max

    @staticmethod
    def from_rewards(rewards: Sequence[float]) -> "MinMaxNormalizer":
        rewards_min: float = min(rewards)
        rewards_max: float = max(rewards)

        return MinMaxNormalizer(
            reward_min=rewards_min,
            reward_max=rewards_max,
        )

    def normalize_reward(self, reward: float) -> float:
        return (reward - self.reward_min) / (self.reward_max - self.reward_min)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name(),
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
        }

    @staticmethod
    def from_dict(_dict: Dict[str, Any]) -> "MinMaxNormalizer":
        reward_min = _dict["reward_min"]
        reward_max = _dict["reward_max"]
        return MinMaxNormalizer(
            reward_min=reward_min,
            reward_max=reward_max,
        )


class DoNothingNormalizer(RewardNormalizer):
    def __init__(self):
        pass

    @staticmethod
    def from_rewards(rewards: Sequence[float]) -> "DoNothingNormalizer":
        return DoNothingNormalizer()

    def normalize_reward(self, reward: float) -> float:
        return reward

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name()}

    @staticmethod
    def from_dict(_dict: Dict[str, Any]) -> "DoNothingNormalizer":
        return DoNothingNormalizer()


class Times1000(RewardNormalizer):
    def __init__(self):
        pass

    @staticmethod
    def from_rewards(rewards: Sequence[float]) -> "Times1000":
        return Times1000()

    def normalize_reward(self, reward: float) -> float:
        return reward * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name()}

    @staticmethod
    def from_dict(_dict: Dict[str, Any]) -> "Times1000":
        return Times1000()


class StandardScaleNormalizer(RewardNormalizer):
    def __init__(
        self,
        mean: float,
        std: float,
    ):
        self.mean = mean
        self.std = std

    @staticmethod
    def from_rewards(rewards: Sequence[float]):
        mean: float = assert_not_none(Slist(rewards).average())
        std: float = assert_not_none(Slist(rewards).standard_deviation())
        return StandardScaleNormalizer(
            mean=mean,
            std=std,
        )

    def normalize_reward(self, reward: float) -> float:
        return (reward - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name(),
            "mean": self.mean,
            "std": self.std,
        }

    @staticmethod
    def from_dict(_dict: Dict[str, Any]) -> "StandardScaleNormalizer":
        mean = _dict["mean"]
        std = _dict["std"]
        return StandardScaleNormalizer(
            mean=mean,
            std=std,
        )


class StandardTimes1000Normalizer(StandardScaleNormalizer):
    # https://github.com/huggingface/blog/blob/main/train-decision-transformers.md
    # Says that the implementation of decision transformers scale the rewards by 1000
    def normalize_reward(self, reward: float) -> float:
        return 1000 * super().normalize_reward(reward)


# Enum of normalizers
class NormalizerOptions(str, Enum):
    min_max = "min_max"
    standard_scale = "standard_scale"
    standard_times_1000 = "standard_times_1000"
    do_nothing = "do_nothing"
    times_1000 = "times_1000"


def get_normalizer(normalizer_option: NormalizerOptions) -> Type[RewardNormalizer]:
    if normalizer_option is NormalizerOptions.min_max:
        return MinMaxNormalizer
    elif normalizer_option is NormalizerOptions.standard_scale:
        return StandardScaleNormalizer
    elif normalizer_option is NormalizerOptions.standard_times_1000:
        return StandardTimes1000Normalizer
    elif normalizer_option is NormalizerOptions.do_nothing:
        return DoNothingNormalizer
    elif normalizer_option is NormalizerOptions.times_1000:
        return Times1000
    else:
        assert_never(normalizer_option)
