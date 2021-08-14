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

dirpath = 'Lifetime analysis/dane/'

# # filename = '2Time.dat'
# for filename in ['1Time.dat','2Time.dat']:
#     data = pd.read_csv(dirpath+filename,sep='\t',skiprows=[1,])
#     t, i = np.array(data['Time']),np.array(data['Intensity'])
#     lim = 0.1
#     i = i/i.sum()
#     t,i = t[t<0.01],i[t<0.01]

#     plt.plot(t, i, label=filename)
# plt.legend()
# plt.show()

# # #### SEF
# # fun = lambda x, A, B, C, D: A*np.exp(-B*(x-0.08)**C) + D
# fun = lambda x, A, B, C, D: A*np.exp(-B*(x-0.08)**C) + D
# p0 = [3829.04734517, 9016.84220052, 1, 46.15589228]

#### EXP & 2EXP
# fun = lambda x, A, T, B: A*np.exp(-T*(x-0.08)) + B
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

# fun_rise = lambda x, T0, T, A, B: A*(1-np.exp(-(x-T0)/T))+B

# def fun_rise(x, T0, T, A, B):
#     ret = A*(1-np.exp(-(x-T0)/T))*np.heaviside(x-T0,1)
#     # ret[ret<0]=0
#     # ret += B
#     return ret

# def fun_rise(x, T01, T1, A, T02, T2, B):
#     if T01 == T02:
#         T01 += T02
#         T02 = 0
#     ret = A*(1-T01/(T01-T02)*np.exp(-x/T1)+T02/(T01-T02)*np.exp(-x/T2))/2
#     ret[ret<0]=0
#     ret += B
#     return ret

def fun_rise(x, T1, T2, A, T0):
    if T1 == T2:
        T1 += T2
        T2 = 0
    x=x-T0
    ret = A*(1-T1/(T1-T2)*np.exp(-x/T1)+T2/(T1-T2)*np.exp(-x/T2))\
        *np.heaviside(x,1)
    # ret[ret<0]=0
    # ret += B
    return ret
p0_rise=None
# p0_rise=(0.001,0.003,350,15) # 1exp
# p0_rise=(0.001,0.001,3900,0.001,0.003,15) # 2exp
p0_rise=(.003,.0032,3900,.001) #iner+hev
dirpath = dirpath + 'dane Marcina/'

fits = []

plt.figure(figsize=(10,5))
filenames = os.listdir(dirpath)
filenames.sort(key = lambda x:-int(x.split(' ')[-1].split('.')[0]))

inds = []


Ts = []
As = []
for filename in filenames:
# for filename in filenames[5:7]:
    print(filename)
    ind = int(filename.split(' ')[-1].split('.')[0])
    inds.append(ind)
    data = pd.read_csv(dirpath+filename,sep='\t',header=None)
    t0, i0 = np.array(data[0]),np.array(data[1])
    # lim = 0.0808
    # t,i = t[t<lim],i[t<lim]
    # lim = 0.08
    # t,i = t[t>lim],i[t>lim]


    lim = 0.02
    # lim = 0.005
    t,i = t0[t0<lim],i0[t0<lim]

    # plt.plot(t,i)
    # if i.max()<1e2: continue

    # plt.figure(figsize=(10,5))
    plt.subplot(121)
    # plt.semilogy(t,i,label=str(ind))
    plt.plot(t,i,label=str(ind))

    popt, pcov = curve_fit(fun_rise,t,i,p0=p0_rise)
    print('--',popt,)
    plt.plot(t,fun_rise(t,*popt))
    Ts.append(popt[[0,1,3]])
    As.append(popt[2])
    # plt.plot(t,fun_rise(t,*p0_rise))
    # plt.plot(t,fun_rise(t,0.001,0.003,350,15))

    # break

    lim = 0.08
    t,i = t0[t0>lim],i0[t0>lim]
    # lim = 0.0808
    lim = 0.082
    t,i = t[t<lim],i[t<lim]
    plt.subplot(122)
    # plt.plot(t,i,label=str(filename.split(' ')[-1].split('.')[0]))
    plt.semilogy(t,i,label=str(filename.split(' ')[-1].split('.')[0]))
    # plt.loglog(t,i,label=str(filename.split(' ')[-1].split('.')[0]))

    try:
        popt, pcov = curve_fit(fun,t,i,p0=p0)
    except:
        break
    p0=popt
    fits.append(popt)
    # A, T, B = popt
    # plt.plot(t,fun(t,*popt))
    plt.semilogy(t,fun(t,*popt))
    # plt.loglog(t,fun(t,*popt))
    print(popt,np.sqrt(np.diag(pcov)))
    print('+/-',3*np.around(np.sqrt(np.diag(pcov))/popt*100,1),'%')
    # break

# plt.figure()
# plt.plot(t,i-fun(t,*popt),'.')

plt.subplot(121);plt.legend()
plt.subplot(122);plt.legend()

# fits = np.array(fits)
# fits[np.abs(fits)==np.inf] = np.nan
# plt.figure()
# plt.plot(inds,fits[:,1],'.')
# plt.plot(inds,fits[:,3],'.')

# plt.figure()
# plt.plot(inds,fits[:,0],'.')
# plt.plot(inds,fits[:,2],'.')

