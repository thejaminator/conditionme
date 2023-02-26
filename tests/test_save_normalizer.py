from pathlib import Path
from typing import List

from conditionme.normalization.normalizer import (
    DoNothingNormalizer,
    RewardNormalizer,
    StandardScaleNormalizer,
    get_normalizer,
    NormalizerOptions,
)


def test_save_normalizer(tmp_path: Path):
    DoNothingNormalizer().save_normalizer(tmp_path)
    normalizer = RewardNormalizer.load_normalizer(tmp_path)
    assert isinstance(normalizer, DoNothingNormalizer)


def test_save_standard_normalizer(tmp_path: Path):
    prev_normalizer = StandardScaleNormalizer(mean=1, std=1)
    prev_normalizer.save_normalizer(tmp_path)
    normalizer = RewardNormalizer.load_normalizer(tmp_path)
    assert isinstance(normalizer, StandardScaleNormalizer)
    assert normalizer.mean == prev_normalizer.mean
    assert normalizer.std == prev_normalizer.std


def test_normalizer_scale():
    normalizer = StandardScaleNormalizer(mean=1, std=1)
    assert normalizer.normalize_reward(1) == 0
    assert normalizer.normalize_reward(2) == 1


def test_normalizer_from_rewards():
    normalizer: StandardScaleNormalizer = StandardScaleNormalizer.from_rewards(
        [1, 2, 3]
    )
    assert normalizer.mean == 2
    assert normalizer.std == 1
    to_normalize: float = 1
    assert normalizer.normalize_reward(to_normalize) == -1
    multiple_to_normalize: List[float] = [1, 2, 3]
    assert normalizer.normalize_rewards(multiple_to_normalize) == [-1, 0, 1]


def test_get_normalizer():
    normalizer_type = get_normalizer(NormalizerOptions.standard_scale)
    assert isinstance(normalizer_type, type(StandardScaleNormalizer))
