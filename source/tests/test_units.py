#!/usr/bin/python3

import numpy as np
import os
import re
import sys
from numpy import load
from matplotlib import pyplot as plt
from datetime import datetime
from .. import solver
from .. import driver

PRECISION=1e-2
ENERGY_CONSERVATION=1e-6

def test_units_prop0():
    from .init_units0 import maindir,cpls,ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    #maindir,cpl,ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling,Nt):
    path=driver.Simulation(maindir,cpls[0],ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings[0],Nt)
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    xmax=[]
    for file in fllst:
        ua=np.load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        indx=np.argmax(uaenv[int(uaenv.shape[0]/2),:])
        xmax.append(x[indx])
    expcdt=[-200+1*np.sqrt(1-0.05)*10*i*dt for i in range(51)]
    dr=[x-y for x,y in zip(xmax,expcdt)]
    from scipy.signal import savgol_filter
    drhat = savgol_filter(dr, 51, 3)
    assert (np.amax(np.abs(drhat)/np.abs(xmax[-1]-xmax[0]))<PRECISION)
    
    fllst2=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ub"):
                fllst2.append(file)
    fllst2.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    xmax2=[]
    for file in fllst2:
        ua=load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        indx=np.argmax(uaenv[int(uaenv.shape[0]/2),:])
        xmax2.append(x[indx])

    expcdt2=[-100+1*np.sqrt(1-0.05/1.2**2)*10*i*dt for i in range(51)]
    dr=[x-y for x,y in zip(xmax2,expcdt2)]
    from scipy.signal import savgol_filter
    drhat = savgol_filter(dr, 51, 3)
    assert (np.amax(np.abs(drhat)/np.abs(xmax2[-1]-xmax2[0]))<PRECISION)

    en1=np.abs(np.loadtxt(path+'/energy1.txt'))
    en2=np.abs(np.loadtxt(path+'/energy2.txt'))
    assert np.abs((np.amax(en1)-np.amin(en1))/np.amax(en1)<ENERGY_CONSERVATION)
    assert np.abs((np.amax(en2)-np.amin(en2))/np.amax(en2)<ENERGY_CONSERVATION)


PRECISION=1e-2

def test_units_prop1():
    from .init_units1 import maindir,cpls,ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.Simulation(maindir,cpls[0],ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings[0],Nt)
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    xmax=[]
    for file in fllst:
        ua=np.load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        indx=np.argmax(uaenv[int(uaenv.shape[0]/2),:])
        xmax.append(x[indx])
    expcdt=[-200+1*np.sqrt(1-0.5)*10*i*dt for i in range(51)]
    dr=[x-y for x,y in zip(xmax,expcdt)]
    from scipy.signal import savgol_filter
    drhat = savgol_filter(dr, 51, 3)
    assert (np.amax(np.abs(drhat)/np.abs(xmax[-1]-xmax[0]))<PRECISION)

    fllst2=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ub"):
                fllst2.append(file)
    fllst2.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    xmax2=[]
    for file in fllst2:
        ua=load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        indx=np.argmax(uaenv[int(uaenv.shape[0]/2),:])
        xmax2.append(x[indx])

    expcdt2=[-200+1*np.sqrt(1-0.5/(1+np.sqrt(0.5))**2)*10*i*dt for i in range(51)]
    dr=[x-y for x,y in zip(xmax2,expcdt2)]
    from scipy.signal import savgol_filter
    drhat = savgol_filter(dr, 51, 3)
    assert (np.amax(np.abs(drhat)/np.abs(xmax2[-1]-xmax2[0]))<PRECISION)


def test_units_prop2():
    from .init_units2 import maindir,cpls,ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.Simulation(maindir,cpls[0],ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings[0],Nt)
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    xmax=[]
    ymax=[]

    for file in fllst:
        ua=np.load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        uamx=np.amax(uaenv)
        indx=np.where(np.abs(uaenv-uamx)<1e-6)
        xmax.append(x[indx[1][0]])
        ymax.append(y[indx[0][0]])
   
    xexpcdt=[-200+np.sqrt(1-0.5)*np.cos(35/180*np.pi)*10*i*dt for i in range(51)]
    yexpcdt=[-np.sqrt(1-0.5)*np.sin(35/180*np.pi)*10*i*dt for i in range(51)]
    drx=[x-y for x,y in zip(xmax,xexpcdt)]
    dry=[x-y for x,y in zip(ymax,yexpcdt)]
    from scipy.signal import savgol_filter
    drxhat = savgol_filter(drx, 51, 3)
    dryhat = savgol_filter(dry, 51, 3)
    assert (np.amax(np.abs(drxhat)/np.abs(xmax[-1]-xmax[0]))<PRECISION)
    assert (np.amax(np.abs(dryhat)/np.abs(ymax[-1]-ymax[0]))<PRECISION)

    fllst2=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ub"):
                fllst2.append(file)
    fllst2.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    xmax2=[]
    ymax2=[]
    for file in fllst2:
        ua=load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        uamx=np.amax(uaenv)
        indx=np.where(np.abs(uaenv-uamx)<1e-6)
        xmax2.append(x[indx[1][0]])
        ymax2.append(y[indx[0][0]])

    xexpcdt2=[100+1*np.sqrt(1-0.5/(1+np.sqrt(0.5))**2)*np.cos(120/180*np.pi)*10*i*dt for i in range(51)]
    yexpcdt2=[173.205080757-1*np.sqrt(1-0.5/(1+np.sqrt(0.5))**2)*np.sin(120/180*np.pi)*10*i*dt for i in range(51)]
    drx=[x-y for x,y in zip(xmax2,xexpcdt2)]
    dry=[x-y for x,y in zip(ymax2,yexpcdt2)]
    from scipy.signal import savgol_filter
    drxhat = savgol_filter(drx, 51, 3)
    dryhat = savgol_filter(dry, 51, 3)
    assert (np.amax(np.abs(drxhat)/np.abs(xmax2[-1]-xmax2[0]))<PRECISION)
    assert (np.amax(np.abs(dryhat)/np.abs(ymax2[-1]-ymax2[0]))<PRECISION)
