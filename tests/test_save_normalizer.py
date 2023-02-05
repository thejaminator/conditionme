from pathlib import Path

from conditionme.normalization.normalizer import DoNothingNormalizer, RewardNormalizer


def test_save_normalizer():
    path = Path("test_normalizer")
    DoNothingNormalizer().save_normalizer(path)
    normalizer = RewardNormalizer.load_normalizer(path)
    assert isinstance(normalizer, DoNothingNormalizer)
