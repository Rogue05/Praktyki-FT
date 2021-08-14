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
from numba import jit, njit


ni = 4
Amax = 10*2

# tau = np.random.uniform(1,20,ni)
# A = np.random.uniform(1,Amax,ni)

A = np.ones(4)*5
# tau = np.linspace(3,18,4)
tau = np.linspace(3,28,4)

# # tau = np.array([2,])
# tau = np.linspace(2-1,2+1,100)
# A = np.ones(len(tau))/len(tau)

Nt = 2000
# t = np.linspace(0,4,Nt)
t = np.linspace(0,1,Nt)

i = np.array(np.sum(A*np.exp(-np.matrix(tau).T*t),axis=0))[0]

nampl = .01
i = i+np.random.uniform(-nampl,nampl,i.shape)

ind = 0

ds = 100
# taus = np.linspace(1,20,ds)
taus = np.linspace(1,30,ds)
wyk = -np.matrix(taus).T

# signature = 'float64[:](float64[:],'+('float64,'*ds)[:-1]+')'
signature = 'float64[:](float64[:],float64[:])'
# print(signature)
# @jit(signature, nopython=True)
# @njit(signature)
@njit()
def model(t, *As):
    # # global ind, wyk
    # # global wyk
    # global ind
    # if ind%100==0:
    #     print(ind)
    # ind += 1

    # return np.array(np.sum(np.array(As)*np.exp(wyk*t),axis=0))[0]

    ret = np.zeros(len(t))
    for i in range(len(As)):
        ret = ret+As[i]*np.exp(-taus[i]*t)
    return ret

# print('-firstcalll')
# # model(t, *A)
# print('================')
# import sys;sys.exit()

p0 = np.ones(ds)/ds

from time import time
start = time()
popt, pcov = curve_fit(model,t,i,p0=p0,bounds = (0,np.inf))
print('=================',time()-start)
plt.plot(taus,popt,'.')
plt.plot(tau,A,'.')
plt.show()
plt.plot(t,i)
plt.plot(t,model(t,*popt))
plt.show()

plt.subplot(211)
plt.plot(taus,popt,ms=10)
plt.plot(tau,A,'.',ms=5)
plt.subplot(212)
plt.plot(t,model(t,*popt)-i)

# sp = np.fft.ifft(i)
# plt.plot(sp.real,'.')
# plt.plot(sp.imag,'.')

# # t = np.sin(np.arange(Nt))
# # t = i
# sp = np.fft.fft(i)
# freq = np.fft.fftfreq(i.shape[-1])
# # plt.plot(freq, np.abs(sp))
# plt.plot(freq, sp.real, freq, sp.imag)


# M = []
# tau2 = np.linspace(1,20,20)
# # for ti in tau:
# for ti in tau2:
#     M.append(np.exp(-ti*t))
# M = np.array(M).T

# nA = np.linalg.inv(M.T@M)@M.T@i
# # print(nA)
# ni = np.array(np.sum(nA*np.exp(-np.matrix(tau2).T*t),axis=0))[0]

# plt.plot(tau,A,'.',ms=10)
# # plt.plot(tau,nA,'.',ms=6)
# plt.plot(tau2,nA,'.',ms=6)
# plt.plot(tau2,np.abs(nA),'.',ms=3)
# plt.show()
# plt.subplot(211)
# plt.plot(t,i)
# plt.plot(t,ni)
# plt.subplot(212)
# plt.plot(t,i-ni)