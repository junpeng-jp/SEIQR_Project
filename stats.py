import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np
import pandas as pd


def overwhelmStats(hospitalised, cap, verbose=None, snsExportName=None, show=False):
    if not verbose:
        verbose = 1

    overwhelmDay = hospitalised[0] > cap
    # first day where hospitals are overshelmed
    overwhelmDay = np.argmax(overwhelmDay, axis=1)

    stats = {
        'pNotOverwhelm': sum(overwhelmDay == 0) / len(overwhelmDay),
        'pOverwhelm': sum(overwhelmDay != 0) / len(overwhelmDay)
    }

    if verbose >= 1:
        print(
            "--- Overwhelm Probability ---",
            f'P(not Overwhelm)\t= {stats["pNotOverwhelm"]  * 100:.3f}',  
            f'P(Overwhelm)\t= {stats["pOverwhelm"] * 100:.3f}', sep='\n'
        )

    if sum(overwhelmDay == 0) > 0:
        stats['notO_Removed'] = hospitalised[1][overwhelmDay == 0, -1].mean()
        stats['notO_WithinCap'] = np.fmin(hospitalised[0][overwhelmDay == 0], cap).sum(axis=1).mean()

        if verbose >= 2:
            print(
                "----- No overwhelm -----",
                f'Patients removed from simulation:\t{stats["notO_Removed"]}', 
                f'Patient Hours within Cap:\t{stats["notO_WithinCap"]}', sep='\n'
            )

    if sum(overwhelmDay != 0) > 0:
        stats['o_Removed'] = hospitalised[1][overwhelmDay != 0, -1].mean()
        stats['o_WithinCap'] = np.fmin(hospitalised[0][overwhelmDay != 0], cap).sum(axis=1).mean()
        stats['o_BeyondCap'] = np.fmax(hospitalised[0][overwhelmDay != 0] - cap, 0).sum(axis=1).mean()
        stats['o_UnattendedProp'] = stats['o_BeyondCap'] / (stats['o_WithinCap'] + stats['o_BeyondCap'])

        if verbose >= 2:
            print(
                "----- Overwhelmed -----",
                "--- Patient after {evalDay} Days ---",
                f'Patients removed from simulation:\t{stats["o_Removed"]}',
                f'Patient Hours within Cap:\t{stats["o_WithinCap"]}',
                f'% of unattended Patients:\t{stats["o_UnattendedProp"]}', sep='\n'
            )

    else:
        print("None of the simulations resulted in overwhelming of healthcare!")

    snsPlot = plotOverwhelmDist(overwhelmDay, show=show)

    if snsExportName and snsPlot:
        snsPlot.savefig(snsExportName)

    stats = pd.DataFrame(stats, index=[0])

    return stats


def plotLineCI(x, y, color, alpha=None, seed=None, show=False):
    if seed:
        np.random.seed(seed)

    if not alpha:
        alpha = 0.2

    plt.fill_between(x, y.min(axis=0), y.max(axis=0), color=color, alpha=alpha)

    for v in np.random.choice(range(len(y)), 100, replace=False):
        sns.lineplot(x, y[v], color=color)

    if show:
        plt.show()


def plotCurve(x, data, color=None, alpha=None, seed=None, show=False):
    if seed:
        np.random.seed(seed)
    if not alpha:
        alpha = 1

    assert len(data) == len(color)

    count = 0
    for i in np.random.choice(range(data['S'].shape[0]), 25, replace=False):
        for k, v in data.items():
            sns.lineplot(x, v[i], color=color[count], alpha=alpha)
            count += 1

        count = 0

    if show:
        plt.show()


def plotOverwhelmDist(Y, color=None, alpha=None, show=False):
    if not alpha:
        alpha = 1

    if sum(Y[Y > 0]) == 0:
        return None

    else:
        minDay = min(Y[Y != 0])
        maxDay = max(Y)

        dist = np.bincount(Y)

        fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
        ax = sns.barplot(np.arange(minDay, maxDay+1), dist[minDay:maxDay+1], color=color, alpha=alpha, ax=ax)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xticklabels(np.arange(minDay, maxDay+1, 5), rotation=90, ha='center')

        if show:
            plt.show()

        return fig
