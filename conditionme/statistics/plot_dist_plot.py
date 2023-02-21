from typing import List, TYPE_CHECKING

import seaborn as sns
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from seaborn._core.plot import Plot


def plot_distribution(
    scores: List[float], title: str, xlabel: str, ylabel: str, bins: int = 20
) -> Plot:
    # plots a distribution chart
    # clear seaborn
    sns.reset_orig()
    f, axes = plt.subplots(1)
    # use seaborn style defaults and set the default figure size
    sns.set(rc={"figure.figsize": (8, 8)})
    # use x and y as the data, assign to the variables called x and y
    # pass axes so you don't overwrite the same plots
    # Don't show the KDE line
    plot = sns.histplot(scores, kde=False, ax=axes, stat="probability", bins=bins)
    # set a title for the chart
    plot.set_title(title)

    # set the labels for the x and y axes
    plot.set(xlabel=xlabel, ylabel=ylabel)
    return plot
