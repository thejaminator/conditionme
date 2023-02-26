import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Sequence, Dict, List, Type

from slist import Slist

from conditionme.type_check.utils import assert_not_none, assert_never


class RewardScaler(ABC):
    @staticmethod
    @abstractmethod
    def from_rewards(rewards: Sequence[float]) -> "RewardScaler":
        raise NotImplementedError

    @abstractmethod
    def scale_reward(self, reward: float) -> float:
        raise NotImplementedError

    def scale_rewards(self, rewards: Sequence[float]) -> List[float]:
        return [self.scale_reward(reward) for reward in rewards]

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def save_scaler(self, path: Path) -> None:
        # Create the directory if it does not exist
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "scaler"
        with open(file_path, "w") as f:
            print(f"Saving scaler to {file_path}")
            f.write(self.to_json())

    @classmethod
    def load_scaler(cls, path: Path) -> "RewardScaler":
        file_path = path / "scaler"
        with open(file_path, "r") as f:
            print(f"Loading scaler from {file_path}")
            _dict = json.load(f)
        return cls.create_from_dict(_dict)

    @staticmethod
    def create_from_dict(_dict: Dict[str, Any]) -> "RewardScaler":
        name = _dict["name"]
        if name == MinMaxScaler.name():
            return MinMaxScaler.from_dict(_dict)
        elif name == StandardScaleScaler.name():
            return StandardScaleScaler.from_dict(_dict)
        elif name == DoNothingScaler.name():
            return DoNothingScaler()
        elif name == Times1000.name():
            return Times1000()
        elif name == StandardTimes1000Scaler.name():
            return StandardTimes1000Scaler.from_dict(_dict)
        else:
            raise ValueError(f"Unknown scaler name: {name}")


class MinMaxScaler(RewardScaler):
    # A scaler that will scale the rewards to be between 0 and 1
    def __init__(
        self,
        reward_min: float,
        reward_max: float,
    ):
        self.reward_min = reward_min
        self.reward_max = reward_max

    @staticmethod
    def from_rewards(rewards: Sequence[float]) -> "MinMaxScaler":
        rewards_min: float = min(rewards)
        rewards_max: float = max(rewards)

        return MinMaxScaler(
            reward_min=rewards_min,
            reward_max=rewards_max,
        )

    def scale_reward(self, reward: float) -> float:
        return (reward - self.reward_min) / (self.reward_max - self.reward_min)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name(),
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
        }

    @staticmethod
    def from_dict(_dict: Dict[str, Any]) -> "MinMaxScaler":
        reward_min = _dict["reward_min"]
        reward_max = _dict["reward_max"]
        return MinMaxScaler(
            reward_min=reward_min,
            reward_max=reward_max,
        )


class DoNothingScaler(RewardScaler):
    def __init__(self):
        pass

    @staticmethod
    def from_rewards(rewards: Sequence[float]) -> "DoNothingScaler":
        return DoNothingScaler()

    def scale_reward(self, reward: float) -> float:
        return reward

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name()}

    @staticmethod
    def from_dict(_dict: Dict[str, Any]) -> "DoNothingScaler":
        return DoNothingScaler()


class Times1000(RewardScaler):
    def __init__(self):
        pass

    @staticmethod
    def from_rewards(rewards: Sequence[float]) -> "Times1000":
        return Times1000()

    def scale_reward(self, reward: float) -> float:
        return reward * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name()}

    @staticmethod
    def from_dict(_dict: Dict[str, Any]) -> "Times1000":
        return Times1000()


class StandardScaleScaler(RewardScaler):
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
        return StandardScaleScaler(
            mean=mean,
            std=std,
        )

    def scale_reward(self, reward: float) -> float:
        return (reward - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name(),
            "mean": self.mean,
            "std": self.std,
        }

    @staticmethod
    def from_dict(_dict: Dict[str, Any]) -> "StandardScaleScaler":
        mean = _dict["mean"]
        std = _dict["std"]
        return StandardScaleScaler(
            mean=mean,
            std=std,
        )


class StandardTimes1000Scaler(StandardScaleScaler):
    # https://github.com/huggingface/blog/blob/main/train-decision-transformers.md
    # Says that the implementation of decision transformers scale the rewards by 1000
    def scale_reward(self, reward: float) -> float:
        return 1000 * super().scale_reward(reward)


# Enum of scalers
class ScalerOptions(str, Enum):
    min_max = "min_max"
    standard_scale = "standard_scale"
    standard_times_1000 = "standard_times_1000"
    do_nothing = "do_nothing"
    times_1000 = "times_1000"


def get_scaler(scaler_option: ScalerOptions) -> Type[RewardScaler]:
    if scaler_option is ScalerOptions.min_max:
        return MinMaxScaler
    elif scaler_option is ScalerOptions.standard_scale:
        return StandardScaleScaler
    elif scaler_option is ScalerOptions.standard_times_1000:
        return StandardTimes1000Scaler
    elif scaler_option is ScalerOptions.do_nothing:
        return DoNothingScaler
    elif scaler_option is ScalerOptions.times_1000:
        return Times1000
    else:
        assert_never(scaler_option)
