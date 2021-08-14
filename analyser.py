# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:24:15 2021

@author: Wojtek
"""
import os
import warnings
from inspect import signature, getsource

from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sympy import symbols
from sympy.utilities.lambdify import lambdify, implemented_function

dirpath = 'Lifetime analysis/dane/'

#### EXP & 2EXP
# fun = lambda x, A, T, B: A*np.exp(-T*(x-0.08)) + B
fun = lambda x, A1, T1, A2, T2: A1*np.exp(-T1*(x-0.08)) + A2*np.exp(-T2*(x-0.08))
# fun = lambda x, A1, T1, A2, T2, B: A1*np.exp(-T1*(x-0.08)) + A2*np.exp(-T2*(x-0.08)) + B
# fun = lambda x, A1, T1, A2, T2, A3, T3, B: \
#     A1*np.exp(-T1*(x-0.08)) + \
#     A2*np.exp(-T2*(x-0.08)) + \
#     A3*np.exp(-T3*(x-0.08)) + B
fun.p0 = None
# fun = lambda x, A, T: A*np.exp(np.exp(-T*(x-0.08)))
# fun = lambda x, A, T, B: A*np.exp(-T*(x-0.08))+B
# p0=(527.94641168, 5083.02901712)
p0=None

def fun_rise(x, T1, T2, A, T0):
    if T1 == T2:
        T1 += T2
        T2 = 0
    x=x-T0
    ret = A*(1-T1/(T1-T2)*np.exp(-x/T1)+T2/(T1-T2)*np.exp(-x/T2))\
        *np.heaviside(x,1)
    return ret
fun_rise.p0 = (.003,.0032,3900,.001)

p0_rise=None
# p0_rise=(0.001,0.003,350,15) # 1exp
# p0_rise=(0.001,0.001,3900,0.001,0.003,15) # 2exp
p0_rise=(.003,.0032,3900,.001) #iner+hev
dirpath = dirpath + 'dane Marcina/'

fits = []

filenames = os.listdir(dirpath)
filenames.sort(key = lambda x:-int(x.split(' ')[-1].split('.')[0]))

inds = []
Ts,As = [], []

def perform_analysis(t, i, fun):
    popt, pcov = curve_fit(fun,t,i,p0=fun.p0)
    ev = fun(t, *popt)
    r = i - ev
    return popt, np.sqrt(np.diag(pcov)), r, np.sum(r**2/ev)

report = ''

for filename in filenames:
    # plt.figure(figsize=(10,5))
    plt.figure(figsize=(14,7))
    plt.suptitle(filename)
    # print(filename)
    ind = int(filename.split(' ')[-1].split('.')[0])
    inds.append(ind)
    data = pd.read_csv(dirpath+filename,sep='\t',header=None)
    t0, i0 = np.array(data[0]),np.array(data[1])

    lim = 0.02
    t,i = t0[t0<lim],i0[t0<lim]

    plt.subplot(221)
    plt.plot(t,i,label=str(ind))
    plt.ylabel('counts')
    plt.title('rise')

    popt, stderr, r, chisq = perform_analysis(t,i,fun_rise)
    plt.plot(t,fun_rise(t,*popt))
    plt.subplot(425)
    plt.plot(t,r)
    plt.ylabel('residuals')

    lim = 0.08
    t,i = t0[t0>lim],i0[t0>lim]
    lim = 0.082
    t,i = t[t<lim],i[t<lim]

    plt.subplot(222)
    plt.semilogy(t,i,label=str(filename.split(' ')[-1].split('.')[0]))

    popt, stderr, r, chisq = perform_analysis(t,i,fun)

    Ts.append(popt[[1,3]])
    As.append(popt[[0,2]])

    fits.append(popt)
    plt.semilogy(t,fun(t,*popt))
    plt.ylabel('counts')
    plt.title('decay')

    plt.subplot(426)
    plt.plot(t,r)
    plt.ylabel('residuals')

    print(popt,stderr)
    plt.tight_layout()

    # print(getsource(fun))

    savepath = 'tmp/'+filename+'.png'
    capt = ''
    iargs = str(signature(fun))[1:-1].split(', ')[1:]

    plt.savefig(savepath)
    report = report + """
\\section{"""+filename+"""}
Funkcja: \\\\"""+getsource(fun)+"""
\\begin{figure}[H]
	\\begin{center}
        \\noindent\makebox[\textwidth]{%
		\\includegraphics[width=20cm]{"""+savepath.split('/')[-1]+"""}}
		\\caption{"""+capt+"""}
	\\end{center}
\\end{figure}

\\begin{center}
    \\begin{tabular}{|"""+' r |'*(len(iargs)+1)+"""}
    \\hline
    Name & """+' & '.join(iargs)+""" \\\\
    \\hline
    Value & """+' & '.join(np.around(popt,2).astype(str))+""" \\\\
    \\hline
    Stderr & """+' & '.join(np.around(stderr,2).astype(str))+""" \\\\
    \\hline
    as \% & """+'\% & '.join((np.around(stderr/popt*100,2)).astype(str))+"""\% \\\\
    \\hline
    \\end{tabular}
\\end{center}
"""

    # import sys; sys.exit()

    plt.show()
    # break



plt.semilogy(inds,1/np.array(Ts),'.')
plt.legend(['T1','T2'])
plt.ylabel('$\\tau$=1/T')
plt.xlabel('index')
plt.savefig('tmp/alltau.png')
plt.show()

plt.semilogy(inds,As,'.');plt.ylim([1,1e4])
plt.legend(['A1','A2'])
plt.ylabel('count')
plt.xlabel('index')
plt.savefig('tmp/alla.png')
plt.show()

report = report+"""
\\section{Summary}
\\begin{figure}[H]
	\\begin{center}
        \\noindent\makebox[\textwidth]{%
		\\includegraphics[width=10cm]{alltau.png}}
		\\caption{}
	\\end{center}
	\\begin{center}
        \\noindent\makebox[\textwidth]{%
		\\includegraphics[width=10cm]{alla.png}}
		\\caption{}
	\\end{center}
\\end{figure}
"""

report = """
\\documentclass{article}
\\usepackage{graphicx}
\\usepackage{subfigure}
\\usepackage{psfrag}
\\usepackage{float}
\\begin{document}
"""+report+"""
\\end{document}
"""

with open('tmp/report.tex','w') as file:
    file.write(report)

import subprocess
ret = subprocess.run(
    'pdflatex.exe -synctex=1 -interaction=nonstopmode --shell-escape .\\report.tex',
    capture_output=True,
    cwd='tmp',
    shell=True)
print(ret.stdout.decode())
