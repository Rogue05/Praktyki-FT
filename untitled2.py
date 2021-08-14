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
dirpath = dirpath + 'dane Marcina/'

plt.figure(figsize=(10,5))
filenames = os.listdir(dirpath)
filenames.sort(key = lambda x:-int(x.split(' ')[-1].split('.')[0]))

inds = []
ret = []
t12 = []
stds = []
avgs = []
for filename in filenames:
# for filename in filenames[5:7]:
    print(filename)
    ind = int(filename.split(' ')[-1].split('.')[0])
    inds.append(ind)
    data = pd.read_csv(dirpath+filename,sep='\t',header=None)
    t0, i0 = np.array(data[0]),np.array(data[1])
    # plt.plot(t0,i0)
    w = 100
    i = np.convolve(i0, np.ones(w), 'valid') / w
    # i-=i.min()-1
    i-=np.average(i0[-1000:])
    # plt.semilogy(t0[:-w+1],i,'.',label=str(ind),ms=.1)
    plt.semilogy(t0[:-w+1],i,label=str(ind))

    # w = len(t0)//100
    # ret.append(np.convolve(i0, np.ones(w), 'valid').max() / w)
    ret.append(i.max())
    # t12.append(t0[(i>ret[-1]*.9).argmax()])
    # t12.append(t0[(i>ret[-1]*.5).argmax()])
    t12.append(t0[(i>(i.max()-i.min())/2).argmax()])
    # break
    stds.append(np.std(i0[len(i0)//4:3*len(i0)//4]))
    avgs.append(np.average(i0[len(i0)//4:3*len(i0)//4]))

# plt.ylim([.2,1e4])
plt.legend()
plt.show()

plt.semilogy(inds,ret,'.')
plt.title('Max I')
plt.show()

# plt.plot(inds,t12,'.')
plt.semilogy(inds,t12,'.')
plt.title('$t_{1/2}$')
