import numpy as np
from scipy.integrate import odeint

def seirODE(Y, t, par):
    N = par['N']
    s = par['infect']
    g = par['recovery']
    b = par['R0'] * (par['death'] + s) * (par['death'] + g) / s  # contact rate
    m = par['birth']
    v = par['death']


    S = Y[0]
    E = Y[1]
    I = Y[2]
    R = Y[3]

    dydt = [-b*I*S/N + m*N - v*S]
    dydt.append(b*I*S/N - (s+v)*E)
    dydt.append(s*E - g*I - v*I)
    dydt.append(g*I - v*R)

    return dydt

def seiqrODE(Y, t, par):
    N = par['N']
    s = par['infect']
    g = par['recovery']
    b = par['R0'] * (par['death'] + s) * (par['death'] + g) / s  # contact rate
    m = par['birth']
    v = par['death']
    iD = par['iD']


    S = Y[0]
    E = Y[1]
    I = Y[2]
    Q = Y[3]
    R = Y[4]

    dydt = [-b*I*S/N + m*N - v*S]
    dydt.append(b*I*S/N - (s+v)*E)
    dydt.append(s*E - iD*I - v*I)
    dydt.append(iD*I - g*Q - v*Q)
    dydt.append(g*Q - v*R)

    return dydt