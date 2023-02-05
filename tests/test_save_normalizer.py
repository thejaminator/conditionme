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
    StandardScaleNormalizer(mean=1, std=1).save_normalizer(path)
    normalizer = RewardNormalizer.load_normalizer(path)
    assert isinstance(normalizer, StandardScaleNormalizer)
