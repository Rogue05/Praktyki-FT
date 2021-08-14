# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:24:15 2021

@author: Wojtek
"""
import os
import warnings

from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


#### EXP & 2EXP
# fun = lambda x, A, T, B: A*np.exp(-T*x) + B
# fun = lambda x, A1, T1, A2, T2: A1*np.exp(-T1*(x-0.08)) + A2*np.exp(-T2*(x-0.08))
# fun = lambda x, A1, T1, A2, T2, B: A1*np.exp(-T1*(x-0.08)) + A2*np.exp(-T2*(x-0.08)) + B
fun = lambda x, A1, T1, A2, T2, A3, T3, B: \
    A1*np.exp(-T1*(x-0.08)) + \
    A2*np.exp(-T2*(x-0.08)) + \
    A3*np.exp(-T3*(x-0.08)) + B

# fun = lambda x, A, T: A*np.exp(np.exp(-T*(x-0.08)))
# fun = lambda x, A, T, B: A*np.exp(-T*(x-0.08))+B
# p0=(527.94641168, 5083.02901712)
p0=None

# tau = np.array([1, 2, 3])
# A = np.array([5, 6, 7])

# tau = np.array([1, 2, 3, 4])
# A = np.array([5, 6, 7, 8])

ni = 2
tau = np.random.uniform(1,20,ni)
A = np.random.uniform(1,20,ni)

Nt = 2000
t = np.linspace(0,1,Nt)

i = np.array(np.sum(A*np.exp(-np.matrix(tau).T*t),axis=0))[0]

nampl = .001
i = i+np.random.uniform(-nampl,nampl,i.shape)

M = []
tau2 = np.linspace(1,20,20)
# for ti in tau:
for ti in tau2:
    M.append(np.exp(-ti*t))
M = np.array(M).T

nA = np.linalg.inv(M.T@M)@M.T@i
# print(nA)
ni = np.array(np.sum(nA*np.exp(-np.matrix(tau2).T*t),axis=0))[0]

plt.plot(tau,A,'.',ms=10)
# plt.plot(tau,nA,'.',ms=6)
plt.plot(tau2,nA,'.',ms=6)
plt.plot(tau2,np.abs(nA),'.',ms=3)
plt.show()
plt.subplot(211)
plt.plot(t,i)
plt.plot(t,ni)
plt.subplot(212)
plt.plot(t,i-ni)
# plt.plot(i-ni)

# popt, pcov = curve_fit(fun,t,i,p0=p0)
# print(A,tau)
# print(popt)

# plt.subplot(211)
# plt.plot(t,i,label='i')
# plt.plot(t, fun(t,*popt),label='curve_fit')
# # plt.plot(t, fun(t,nA[0],tau[0],nA[1],tau[1],nA[2],tau[2],0),label='minsq')
# plt.legend()

# plt.subplot(212)
# plt.plot(t,i-fun(t,*popt))
# # plt.subplot(313)
# # plt.plot(t,i-fun(t,nA[0],tau[0],nA[1],tau[1],nA[2],tau[2],0))

# plt.tight_layout()
