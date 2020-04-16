import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np
import pandas as pd


def overwhelmStats(hospitalised, cap, threshold, evalDay, verbose = None, show=False):
    if not verbose:
        verbose = 1

    overwhelmDay = hospitalised[0] > cap
    overwhelmDay = np.argmax(overwhelmDay, axis=1) # first day where hospitals are overshelmed

    overwhelm = overwhelmDay[(overwhelmDay != 0) & (overwhelmDay <= evalDay)]
    
    stats = {
        ('threshold',): threshold,
        ('probability', 'notOverwhelm'): 1 - (len(overwhelm) / len(overwhelmDay)),
        ('probability', 'below'): sum(overwhelm < threshold) / len(overwhelmDay),
        ('probability', 'above'): sum(overwhelm >= threshold) / len(overwhelmDay)
    }

    if verbose >= 1:
        print(  
            "--- Overwhelm Probability ---",
            f'P(not Overwhelm)\t= {stats[("probability", "notOverwhelm")]  * 100:.3f}',  
            f'P(Overwhelm < {threshold})\t= {stats[("probability", "below")] * 100:.3f}',
            f'P(Overwhelm >= {threshold})\t= {stats[("probability", "above")] * 100:.3f}\n', sep='\n'
        )

    if sum(overwhelmDay == 0) > 0:
        stats[('good', 'inHospital')] = hospitalised[0][overwhelmDay == 0, evalDay].mean(),
        stats[('good', 'removed')] = hospitalised[1][overwhelmDay == 0, evalDay].mean(),
        stats[('good', 'withinCap')] = np.fmin(hospitalised[0][overwhelmDay == 0, :evalDay + 1], cap).sum(axis=1).mean()

        
        if verbose >= 2:
            print(
                "----- No overwhelm -----",
                "--- Patient after {evalDay} Days ---",
                f'Patients still in hospital:\t{stats[("good", "inHospital")]}',
                f'Patients removed from simulation:\t{stats[("good","removed")]}',
                "--- Patient hours ---",
                f'Mean Patient hours within Healthcare Cap:\t{stats[("good", "withinCap")]}', sep='\n'
            )

    if len(overwhelm) > 0:
        stats[('bad', 'inHospital')] = hospitalised[0][overwhelmDay != 0, evalDay].mean()
        stats[('bad', 'removed')] = hospitalised[1][overwhelmDay != 0, evalDay].mean()
        stats[('bad', 'withinCap')] = np.fmin(hospitalised[0][overwhelmDay != 0, :evalDay + 1], cap).sum(axis=1).mean()
        stats[('bad', 'beyondCap')] = np.fmax(hospitalised[0][overwhelmDay != 0, :evalDay + 1] - cap, 0).sum(axis=1).mean()
        
        if verbose >= 2:
            print(
                "----- Overwhelmed -----",
                "--- Patient after {evalDay} Days ---",
                f'Patients still in hospital:\t{stats[("bad", "inHospital")]}',
                f'Patients removed from simulation:\t{stats[("bad","removed")]}',
                "--- Patient hours ---",
                f'Mean Patient hours within Healthcare Cap:\t{stats[("bad", "withinCap")]}',
                f'Mean Patient hours beyond Healthcare Cap:\t{stats[("bad", "beyondCap")]}\n', sep='\n'
            )


    if len(overwhelm) == 0:
        print("None of the simulations resulted in overwhelming of healthcare!")

    else:
        plotOverwhelmDist(overwhelmDay, show=show)

    stats = pd.DataFrame(stats)
    stats.columns = pd.MultiIndex.from_tuples(stats.columns)

    return stats


def plotLineCI(x, y, color, alpha = None, seed = None, show=False):
    if seed:
        np.random.seed(seed)

    if not alpha:
        alpha = 0.2

    plt.fill_between(x, y.min(axis=0), y.max(axis=0), color = color, alpha=alpha)

    for v in np.random.choice(range(len(y)), 100, replace=False):
        sns.lineplot(x, y[v], color = color)

    if show:
        plt.show()

def plotCurve(x, data, color = None, alpha = None, seed = None, show = False):
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


def plotOverwhelmDist(Y, color = None, alpha = None, show = False):
    if not alpha:
        alpha = 1
    
    minDay = min(Y[Y != 0])
    maxDay = max(Y)

    dist = np.bincount(Y)

    ax = sns.barplot(np.arange(minDay, maxDay+1), dist[minDay:maxDay+1], color=color, alpha=alpha)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticklabels(np.arange(minDay, maxDay+1, 5), rotation=90, ha='center')


    if show:
        plt.show()