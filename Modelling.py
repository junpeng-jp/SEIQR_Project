from SEIR import seiqrSimulate
from SEIR_ode import seirODE, seiqrODE
from stats import plotLineCI, plotCurve, plotOverwhelmDist, overwhelmStats

from sklearn.model_selection import ParameterGrid

import numpy as np
import pandas as pd
import seaborn as sns

base = {
    'N': 5612000,
    'birth': 0,
    'death': 0,
    'R0': 2.2,
    'susceptible': 1.0,
    'infect': 1/7,
    'recovery': 1/3,
    'expAscertain': 0,
    'infAscertain': 0
}

base['transmission'] = base['R0'] * (base['death'] + base['infect']) * (base['death'] + base['recovery']) / base['infect']


branch1 = {
    'distFactor': [{61: 1.0}, {61: 0.8, 150: 1.0}, {61: 0.6, 150: 1.0}, {61: 0.2, 90: 1.0}, {61: 0.2, 150: 1.0}],
    'hygieneFactor': [{61: 0.9}, {61: 0.75}],
    'eAscertain': [{61: 0.03}],
    'iAscertain': [{61: 0.15}]
}


branch2 = {
    'distFactor': [{61: 1.0}, {61: 0.8, 150: 1.0}, {61: 0.6, 150: 1.0}, {61: 0.2, 90: 1.0}, {61: 0.2, 150: 1.0}],
    'hygieneFactor': [{61: 0.9}, {61: 0.75}],
    'eAscertain': [{61: 0.03}],
    'iAscertain': [{61: 0.09}]
}


Y0 = [0,10,423,240]
Y0.insert(0, base['N'] - sum(Y0))
t = np.arange(61, 721)
seed = 12345

params = ParameterGrid(branch2)

allStats= []
output = []
runDetail = []

# MonteCarlo
for p in iter(params):
    print("\n".join(['--- Current Run ---'] + [f'{k} : {v}' for k, v in p.items()] +[""]))
    runDetail.append(p)

    data, trace = seiqrSimulate(Y0, t, base, p, nSim=10000, seed = seed)
    print('\n') # for formatting
    output.append(data)

    hospitalised = (data['I'] + data['Q'], data['R'])
    allStats.append(overwhelmStats(hospitalised, 2500, 180-60, 365-60, verbose = 1, show=False))




data = pd.DataFrame(allStats)
data.to_csv('run_data.csv', index=False)

runDetail = pd.DataFrame(runDetail)

output = pd.concat([runDetail, data], axis=1)
output.to_csv('data.csv', index=False)

target = output[0]  # campaign 1
hospitalised = (target['I'] + target['Q'], target['R'])

overwhelmStats(hospitalised, 2500, 180-60, 365-60, verbose = 2, show=True)

cPal = sns.color_palette('muted')
xAxis = np.concatenate(([min(t)-1], t))
plotCurve(xAxis, target, color=cPal[:5], seed=seed, show=True)

plotLineCI(xAxis, hospitalised[0], color=cPal[3], seed=seed, show=True)

plotLineCI(xAxis, hospitalised[1], color=cPal[1], seed=seed, show=True)