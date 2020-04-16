import numpy as np
import progressbar


def seiqrSimulate(Y, t, base, campaign, nSim = None, seed = None):
    if seed:
        np.random.seed(seed)

    if not nSim:
        nSim = 100

    modelGrp = ['S','E','I','Q','R']
    
    YDict = {grp: np.array([[v]] * nSim) for grp, v in zip(modelGrp, Y)}

    # Base parameters
    N = int(base.get('N') * base.get('susceptible'))
    m = base.get('birth')
    v = base.get('death')

    b = base.get('transmission')
    s = base.get('infect')
    g = base.get('recovery')
    eAsc = base.get('expAscertain')
    iAsc = base.get('infAscertain')

    # initial campaign adjustment
    bAdj = 1
    sAdj = 1
    gAdj = 1
    eAscAdj = 1
    iAscAdj = 1

    trace = []
    
    widgets=[
        progressbar.Bar(),
        ' [', progressbar.Percentage(), ']',
    ]


    for day in progressbar.progressbar(t, widgets = widgets):
        bAdj = campaign.get('distFactor', {}).get(day, bAdj)
        hAdj = campaign.get('hygieneFactor', {}).get(day, sAdj)
        gAdj = campaign.get('recoveryFactor', {}).get(day, gAdj)
        eAscAdj = campaign.get('eAscertain', {}).get(day, eAscAdj)
        iAscAdj = campaign.get('iAscertain', {}).get(day, iAscAdj)

        trace.append(b * bAdj)

        nextY = seiqrModel(YDict, N, b * bAdj * hAdj, s , g * gAdj, m, v, eAsc + eAscAdj, iAsc + iAscAdj)
        for grp in YDict.keys():
            YDict[grp] = np.append(YDict[grp], nextY[grp], axis=1)


    return YDict, trace





def seiqrModel(Y, N, b, s, g, m, v, eAsc, iAsc):
    S = Y['S'][:, [-1]]
    E = Y['E'][:, [-1]]
    I = Y['I'][:, [-1]]
    Q = Y['Q'][:, [-1]]
    R = Y['R'][:, [-1]]

    inputShape = S.shape

    nextS = np.random.binomial(S, b*I/N, inputShape)            # Susceptible to infected
    nextE = np.random.binomial(E, s, inputShape)                # Exposed to Infected
    nextEI = np.random.binomial(E - nextE, eAsc, inputShape)    # Exposed to Quarantine
    nextI = np.random.binomial(I, g, inputShape)                # Infected to Recovered
    nextQI = np.random.binomial(I - nextI, iAsc, inputShape)    # Infected to Quarantine
    nextQ = np.random.binomial(Q, g, inputShape)                # Quarantine to Recovered
    
    nextY = {
        'S': np.fmax(S - nextS + m*N - v * S, 0),
        'E': np.fmax(E + nextS - nextE - nextEI - v * E, 0),
        'I': np.fmax(I + nextE - nextI - nextQI - v * I, 0),
        'Q': np.fmax(Q + nextEI + nextQI - nextQ - v * Q , 0),
        'R': np.fmax(R + nextI + nextQ - v * R, 0)
    }

    return nextY