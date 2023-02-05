import dataclasses
from typing import List

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import _result_classes, pearsonr
from scipy.stats._common import ConfidenceInterval


@dataclasses.dataclass
class ScatterplotResults:
    figure: Figure
    correlation: float
    p_value: float
    confidence_level: float
    upper_correlation_bound: float
    lower_correlation_bound: float


def plot_scatterplot_and_correlation(
    x: List[float], y: List[float], title: str, xlabel: str, ylabel: str
) -> ScatterplotResults:
    # clear seaborn
    sns.reset_orig()
    f, axes = plt.subplots(1)
    # use seaborn style defaults and set the default figure size
    sns.set(rc={"figure.figsize": (8, 8)})
    # use x and y as the data, assign to the variables called x and y
    # use the function regplot to make a scatterplot
    # color the scatterplot points blue
    # pass axes so you don't overwrite the same plots
    plot = sns.regplot(x=x, y=y, color="b", line_kws={"color": "red"}, ax=axes)
    # add a (1, 1) line to show perfect correlation
    plot.plot([0, 1], [0, 1], transform=plot.transAxes, ls="--", c=".3")
    # Calculate the correlation coefficient between x and y
    pearson: _result_classes.PearsonRResult = pearsonr(
        x=x,
        y=y,
    )
    confidence_level = 0.95
    confidence_interval: ConfidenceInterval = pearson.confidence_interval(
        confidence_level=confidence_level
    )
    lower_bound = confidence_interval[0]
    upper_bound = confidence_interval[1]
    correlation: float = pearson.statistic
    pvalue: float = pearson.pvalue

    # set a title for the regplot
    title_with_statistics = f"{title} Correlation: {correlation:.2f}, [{lower_bound:.2f}, {upper_bound:.2f}]"
    plot.figure.suptitle(title_with_statistics)
    # set the labels for the x and y axes
    plot.set(xlabel=xlabel, ylabel=ylabel)
    # set the x and y axis to (0, 1)
    plot.set(xlim=(0, 1), ylim=(0, 1))
    figure = plot.figure

    return ScatterplotResults(
        figure=figure,
        correlation=correlation,
        p_value=pvalue,
        confidence_level=confidence_level,
        upper_correlation_bound=upper_bound,
        lower_correlation_bound=lower_bound,
    )
