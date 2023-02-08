from pathlib import Path

from conditionme.normalization.normalizer import (
    DoNothingNormalizer,
    RewardNormalizer,
    StandardScaleNormalizer,
)


def test_save_normalizer():
    path = Path("test_normalizer")
    DoNothingNormalizer().save_normalizer(path)
    normalizer = RewardNormalizer.load_normalizer(path)
    assert isinstance(normalizer, DoNothingNormalizer)


def test_save_standard_normalizer():
    path = Path("test_normalizer")
    prev_normalizer = StandardScaleNormalizer(mean=1, std=1)
    prev_normalizer.save_normalizer(path)
    normalizer = RewardNormalizer.load_normalizer(path)
    assert isinstance(normalizer, StandardScaleNormalizer)
    assert normalizer.mean == prev_normalizer.mean
    assert normalizer.std == prev_normalizer.std


def test_normalizer_scale():
    normalizer = StandardScaleNormalizer(mean=1, std=1)
    assert normalizer.normalize_reward(1) == 0
    assert normalizer.normalize_reward(2) == 1
