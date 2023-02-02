import dataclasses
from typing import Optional, Sequence

from slist import Slist, identity


@dataclasses.dataclass
class DistributionStatistic:
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    one_percentile: Optional[float]
    five_percentile: Optional[float]
    twenty_five_percentile: Optional[float]
    seventy_five_percentile: Optional[float]
    ninety_five_percentile: Optional[float]
    ninety_nine_percentile: Optional[float]
    count: int


def calculate_distribution_statistics(
    dist: Sequence[float],
) -> DistributionStatistic:
    distribution = Slist(dist)
    return DistributionStatistic(
        min=distribution.min_by(identity),
        max=distribution.max_by(identity),
        mean=distribution.average(),
        median=distribution.median_by(identity),
        std=distribution.standard_deviation(),
        one_percentile=distribution.percentile_by(identity, 0.01),
        five_percentile=distribution.percentile_by(identity, 0.05),
        twenty_five_percentile=distribution.percentile_by(identity, 0.25),
        seventy_five_percentile=distribution.percentile_by(identity, 0.75),
        ninety_five_percentile=distribution.percentile_by(identity, 0.95),
        ninety_nine_percentile=distribution.percentile_by(identity, 0.99),
        count=len(distribution),
    )
