from pathlib import Path
from typing import List

from conditionme.scaling.scaler import (
    DoNothingScaler,
    RewardScaler,
    StandardScaleScaler,
    get_scaler,
    ScalerOptions,
)


def test_save_scaler(tmp_path: Path):
    DoNothingScaler().save_scaler(tmp_path)
    scaler = RewardScaler.load_scaler(tmp_path)
    assert isinstance(scaler, DoNothingScaler)


def test_save_standard_scaler(tmp_path: Path):
    prev_scaler = StandardScaleScaler(mean=1, std=1)
    prev_scaler.save_scaler(tmp_path)
    scaler = RewardScaler.load_scaler(tmp_path)
    assert isinstance(scaler, StandardScaleScaler)
    assert scaler.mean == prev_scaler.mean
    assert scaler.std == prev_scaler.std


def test_scaler_scale():
    scaler = StandardScaleScaler(mean=1, std=1)
    assert scaler.scale_reward(1) == 0
    assert scaler.scale_reward(2) == 1


def test_scaler_from_rewards():
    scaler: StandardScaleScaler = StandardScaleScaler.from_rewards(
        [1, 2, 3]
    )
    assert scaler.mean == 2
    assert scaler.std == 1
    to_normalize: float = 1
    assert scaler.scale_reward(to_normalize) == -1
    multiple_to_normalize: List[float] = [1, 2, 3]
    assert scaler.scale_rewards(multiple_to_normalize) == [-1, 0, 1]


def test_get_scaler():
    scaler_type = get_scaler(ScalerOptions.standard_scale)
    assert isinstance(scaler_type, type(StandardScaleScaler))
