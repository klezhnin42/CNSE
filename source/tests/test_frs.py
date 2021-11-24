#!/usr/bin/python3

import numpy as np
import os
import re
import sys
from numpy import load
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime
from .. import solver
from .. import driver

FRS_PRECISION=5e-2
ENERGY_CONSERVATION=1e-6

def test_frs0():
    from .init_frs0 import maindir,cpls,ua0,ub0,vga,vgb,cvph1,cvph2,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.Simulation(maindir,cpls[0],ua0,ub0,vga,vgb,cvph1,cvph2,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings[0],Nt)
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    ymax=[]
    for file in fllst:
        ua=np.load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        ymax.append(np.amax(uaenv))

    # define plasma frequency
    wpew0 = w2w1 - 1.0
    # define plasma density
    nencr=wpew0**2
    # define pump field amplitude
    a1 = np.amax(ub0)
    # define time axis
    times=np.linspace(0,500,51)
    # calc FRS growth rate
    gFRS = 0.25*(1.0/(1.0+wpew0))**1.5*(nencr)**0.75*a1
    # calculate expected linearized growth
    expected_grwth = [1.0 + gFRS*t for t in times] 
    # normalize evolution of seed max amplitude
    ymax = [y/ymax[0] for y in ymax]
    # relative difference
    dd = [(x-y)/x for x,y in zip(expected_grwth,ymax)]
    # smoothen the array above
    ddsmooth = savgol_filter(dd, 51, 5)
    assert np.amax(np.abs(ddsmooth)) < FRS_PRECISION
    
    # check energy conservation
    en1=np.abs(np.loadtxt(path+'/energy1.txt'))
    en2=np.abs(np.loadtxt(path+'/energy2.txt'))
    entot=en1+en2
    assert np.abs((np.amax(entot)-np.amin(entot))/np.amax(entot)<ENERGY_CONSERVATION)


