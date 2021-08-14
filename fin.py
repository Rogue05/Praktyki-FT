# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:18:51 2021

@author: Wojtek
"""
import os
import re

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

def process_name(filename):
    filename = os.path.split(filename)[-1]
    date, sample_nr, power = re.split('Concentration|Power|Sample',filename[:-4])
    return date, sample_nr, power

class Exp:
    fun = lambda t, A, T, B: A*np.exp(-t/T)+B

    def get_p0(t,y):
        B = y.min()
        A = y.max()-y.min()
        T = (t.max()-t.min())/3
        return A, T, B

class Exp_rise:
    fun = lambda t, A, T, B: A*(1-np.exp(-t/T))+B

    def get_p0(t,y):
        B = y.min()
        A = y.max()-y.min()
        T = (t.max()-t.min())/3
        return A, T, B

class Exp2:
    fun = lambda t, A1, T1, A2, T2, B: A1*np.exp(-t/T1)+ A2*np.exp(-t/T2)+B

    def get_p0(t,y):
        B = y.min()
        A1 = (y.max()-y.min())/2
        A2 = A1
        T1 = (t.max()-t.min())/3
        T2 = T1*2
        return A1, T1, A2, T2, B

class Exp2_rise:
    fun = lambda t, A1, T1, A2, T2, B: A1*(1-np.exp(-t/T1))+ A2*(1-np.exp(-t/T2))+B

    def get_p0(t,y):
        B = y.min()
        A1 = (y.max()-y.min())/2
        A2 = A1
        T1 = (t.max()-t.min())/3
        T2 = T1*2
        return A1, T1, A2, T2, B

class ExpS:
    fun = lambda t, A, B, C, D: A*np.exp(-t**C/B) + D

    def get_p0(t,y):
        D = y.min()
        A = y.max()-y.min()
        C = 1
        B = (t.max()-t.min())/10
        return A, B, C, D

class ExpS_rise:
    fun = lambda t, A, B, C, D: A*(1-np.exp(-t**C/B)) + D

    def get_p0(t,y):
        D = y.min()
        A = y.max()-y.min()
        C = 1
        B = (t.max()-t.min())/10
        return A, B, C, D


class ExpSpec:
    def __init__(self, taumin, taumax, N):
        self.tau = np.logspace(np.log10(taumin),np.log10(taumax),N)

    def fun(self, t, *As):
        ret = np.zeros(len(t))
        for i in range(len(As)):
            ret = ret+As[i]*np.exp(-t/self.tau[i])
        return ret

    def get_p0(self, t,y):
        return np.ones(len(self.tau))*np.average(y[:10])/len(self.tau)

def process_data(t,y,t_start,t_end,model):
    di = np.logical_and(t>t_start,t<t_end)
    dt, dy = t[di], y[di]
    p0 = model.get_p0(dt, dy)

    popt, pcov = curve_fit(model.fun,dt-t_start,dy,p0,bounds = (0,np.inf))
    stderr = np.sqrt(np.diag(pcov))
    r = dy - model.fun(dt-t_start,*popt)

    return popt, stderr, r, dt, dy

def eval_model(model, t, p):
    return model.fun(t,*p)

def process_file(filename, t_start, t_end, model):
    try:
        data = pd.read_csv(filename,
                           sep=';',
                           decimal=',',
                           header=None,
                           skiprows=15)
    except pd.errors.EmptyDataError:
        print('ERROR invalid file',filename)
        return
    except FileNotFoundError:
        print('ERROR missing file',filename)
        return

    t,y = np.array(data[0]),np.array(data[1])
    return process_data(t, y, t_start, t_end, model)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dirpath = 'respotkanieponiedziaek'
    model = ExpSpec(1e-4,1e1,51)

    for filename in os.listdir(dirpath):
        print('=====',filename,process_name(filename))
        date, sample_nr, power = process_name(filename)
        ret = process_file(os.path.join(dirpath,filename),
                           0.025,0.030,
                           model)
        if ret is None: continue
        popt, stderr, r, t, y = ret
        print(stderr)
        plt.show()
        plt.suptitle(filename)
        plt.subplot(211)
        plt.semilogx(model.tau,popt/popt.sum())
        plt.subplot(212)
        plt.plot(t, np.abs(r/y),'.',ms=1)

