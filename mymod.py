# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:18:51 2021

@author: Wojtek
"""
import os
import re
import warnings

from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

def process_name(filename):
    # date = filename[:10]
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

def process_data(t,y,t_start,t_end,model):
    di = np.logical_and(t>t_start,t<t_end)
    dt, dy = t[di], y[di]
    plt.plot(dt,dy,label='decay')
    # plt.legend()

    p0 = model.get_p0(dt, dy)
    plt.plot(dt,model.fun(dt-t_start,*p0))


    popt, pcov = curve_fit(model.fun,dt-t_start,dy,p0)
    stderr = np.sqrt(np.diag(pcov))
    r = dy - model.fun(dt-t_start,*popt)
    plt.plot(dt,model.fun(dt-t_start,*popt))
    plt.show()

    return popt, stderr, r


def process_file(filename, t_start, t_end, model):
    try:
        data = pd.read_csv(filename,
                           sep=';',
                           decimal=',',
                           header=None,
                           skiprows=15)
    except pd.errors.EmptyDataError:
        print('ERROR invalid file',filename)
        # continue
        return
    except FileNotFoundError:
        print('ERROR missing file',filename)
        # continue
        return

    t,y = np.array(data[0]),np.array(data[1])
    # return process_data(t,y,0.025,0.030,Exp2)
    # return process_data(t,y,0.019,0.025,ExpS_rise)
    return process_data(t, y, t_start, t_end, model)


if __name__ == '__main__':
    dirpath = 'respotkanieponiedziaek'

    for filename in os.listdir(dirpath):
        print(filename,process_name(filename))
        date, sample_nr, power = process_name(filename)
        ret = process_file(os.path.join(dirpath,filename), 0.019,0.025,ExpS_rise)
        if ret is None: continue
        popt, stderr, r = ret
        print(stderr)
        if filename == '28.06.2021Sample208PowerHigh.txt':
            break
