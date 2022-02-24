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

WIDTH_PRECISION=3e-2
ENERGY_CONSERVATION=1e-6

def test_focus0():
    from .init_focus0 import maindir,cpls,ua0,ub0,vga,vgb,cvph1,cvph2,w2w1,wpw1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.Simulation(maindir,cpls[0],ua0,ub0,vga,vgb,cvph1,cvph2,w2w1,wpw1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings[0],Nt)
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    xmax=[]
    ymax=[]
    dw=[]
    for file in fllst:
        ua=np.load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        ymax.append(np.amax(uaenv))
        indx=np.argmax(uaenv[int(uaenv.shape[0]/2),:])
        indx2=np.where(np.abs(uaenv[int(uaenv.shape[0]/2):,indx]-np.amax(uaenv[int(uaenv.shape[0]/2):,indx])/np.exp(1.0))<3e-3)
        indx2r=indx2[0][int(len(indx2[0])/2)]
        xmax.append(x[indx])
        dw.append(np.abs(y[int(uaenv.shape[0]/2)+indx2r]))

    # beam waist & Rayleigh length in physical units
    w0a=2.0
    zRa = np.pi*w0a**2
    #  converting to physical units to compare with theoretical predict.
    dw=[x/2/np.pi for x in dw]
    # same here
    xmax = [x/(2*np.pi) for x in xmax]

    # analytical behavior of the Gaussian beam width
    dwexpctd=[w0a*np.sqrt(1+x**2/zRa**2) for x in xmax] 
    dwdr=[(x-y)/x for x,y in zip(dw,dwexpctd)]
    
    assert np.mean(np.abs(dwdr))<WIDTH_PRECISION

    en1=np.abs(np.loadtxt(path+'/energy1.txt'))
    assert np.abs((np.amax(en1)-np.amin(en1))/np.amax(en1)<ENERGY_CONSERVATION)


def test_focus1():
    from .init_focus1 import maindir,cpls,ua0,ub0,vga,vgb,cvph1,cvph2,w2w1,wpw1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.Simulation(maindir,cpls[0],ua0,ub0,vga,vgb,cvph1,cvph2,w2w1,wpw1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings[0],Nt)
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    xmax=[]
    ymax=[]
    dw=[]
    for file in fllst:
        ua=np.load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        ymax.append(np.amax(uaenv))
        indx=np.argmax(uaenv[int(uaenv.shape[0]/2),:])
        indx2=np.where(np.abs(uaenv[int(uaenv.shape[0]/2):,indx]-np.amax(uaenv[int(uaenv.shape[0]/2):,indx])/np.exp(1.0))<3e-3)
        indx2r=indx2[0][int(len(indx2[0])/2)]
        xmax.append(x[indx])
        dw.append(np.abs(y[int(uaenv.shape[0]/2)+indx2r]))

    # beam waist & Rayleigh length in physical units
    w0a=2.0
    zRa = np.pi*w0a**2*np.sqrt(1-0.8**2)
    #  converting to physical units to compare with theoretical predict.
    dw=[x/2/np.pi for x in dw]
    # same here
    xmax = [x/(2*np.pi) for x in xmax]

    # analytical behavior of the Gaussian beam width
    dwexpctd=[w0a*np.sqrt(1+x**2/zRa**2) for x in xmax]
    dwdr=[(x-y)/x for x,y in zip(dw,dwexpctd)]

    assert np.mean(np.abs(dwdr))<WIDTH_PRECISION

    en1=np.abs(np.loadtxt(path+'/energy1.txt'))
    assert np.abs((np.amax(en1)-np.amin(en1))/np.amax(en1)<ENERGY_CONSERVATION)


    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ub"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    xmax=[]
    ymax=[]
    dw=[]
    for file in fllst:
        ua=np.load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        ymax.append(np.amax(uaenv))
        indx=np.argmax(uaenv[int(uaenv.shape[0]/2),:])   
        indx2=np.where(np.abs(uaenv[int(uaenv.shape[0]/2):,indx]-np.amax(uaenv[int(uaenv.shape[0]/2):,indx])/np.exp(1.0))<3e-3)
        indx2r=indx2[0][int(len(indx2[0])/2)]
        xmax.append(x[indx])
        dw.append(np.abs(y[int(uaenv.shape[0]/2)+indx2r]))

    # beam waist & Rayleigh length in physical units
    w0a=4.0
    zRa = np.pi*w0a**2*np.sqrt(1-0.8**2/4)/0.5
    #  converting to physical units to compare with theoretical predict.
    dw=[x/2/np.pi for x in dw]
    # same here
    xmax = [x/(2*np.pi) for x in xmax]

    # analytical behavior of the Gaussian beam width
    dwexpctd=[w0a*np.sqrt(1+x**2/zRa**2) for x in xmax]
    dwdr=[(x-y)/x for x,y in zip(dw,dwexpctd)]

    assert np.mean(np.abs(dwdr))<WIDTH_PRECISION

    en1=np.abs(np.loadtxt(path+'/energy1.txt'))
    assert np.abs((np.amax(en1)-np.amin(en1))/np.amax(en1)<ENERGY_CONSERVATION)

    en2=np.abs(np.loadtxt(path+'/energy2.txt'))
    assert np.abs((np.amax(en2)-np.amin(en2))/np.amax(en2)<ENERGY_CONSERVATION)


