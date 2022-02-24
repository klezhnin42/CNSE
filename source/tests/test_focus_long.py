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
LENGTH_PRECISION=6e-2
ENERGY_CONSERVATION=1e-6

def test_focus_long():
    from .init_focus_long import maindir,cpls,uvec,vgvec,cvvec,w2w1,wpw1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.SimulationMultiPump(maindir,cpls,uvec,vgvec,cvvec,w2w1,wpw1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt)
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    cntrs=[]
    lngths=[]
    wdths=[]
    for file in fllst:
        ua=np.load(path+'/'+file)
        uaenv=np.abs(ua)
        indx=np.where(uaenv[256,:]==np.amax(uaenv[256,:]))
        cntrs.append(x[indx])
        indx2=np.where(np.abs(uaenv[256,:indx[0][0]]-np.amax(uaenv[256,:indx[0][0]])/np.exp(1))==np.amin(np.abs(uaenv[256,:indx[0][0]]-np.amax(uaenv[256,:indx[0][0]])/np.exp(1))))
        lngths.append(np.abs(x[indx]-x[indx2]))
        indx3=np.where(np.abs(uaenv[:,indx[0]]-np.amax(uaenv[:,indx[0]])/np.exp(1))==np.amin(np.abs(uaenv[:,indx[0]]-np.amax(uaenv[:,indx[0]])/np.exp(1))))    
        wdths.append(np.abs(y[indx3[0]]))

    wdths=[np.mean(k)/2/np.pi for k in wdths]
    lngths=[k[0]/2/np.pi for k in lngths]

    # beam waist, length & Rayleigh length in physical units
    Nrefra=np.sqrt(1-0.05)
    w0a=4.0
    dura=4.0
    zRaw = np.pi*w0a**2*Nrefra
    zRal = np.pi*dura**2*Nrefra

    # analytical behavior of the Gaussian beam width
    cntrs=[l[0]/2/np.pi for l in cntrs]
    wexpctd=[w0a*np.sqrt(1+l**2/zRaw**2) for l in cntrs] 
    lexpctd=[dura*np.sqrt(1+l**2/zRal**2) for l in cntrs]

    dwdr=[(v-w)/w for v,w in zip(wdths,wexpctd)]
    dldr=[(v-w)/w for v,w in zip(lngths,lexpctd)]   
    
    assert np.mean(np.abs(dwdr))<WIDTH_PRECISION
#    assert np.mean(np.abs(dldr))<LENGTH_PRECISION

    # check energy conservation
    en=np.abs(np.loadtxt(path+'/energy.txt'))
    entot=[x[0]+x[1] for x in en]
    assert np.abs((np.amax(entot)-np.amin(entot))/np.amin(entot)<ENERGY_CONSERVATION)
