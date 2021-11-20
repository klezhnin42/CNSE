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

WIDTH_PRECISION=5e-2
ENERGY_CONSERVATION=1e-6

def test_focus():
    from .init_focus import maindir,cpls,ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.Simulation(maindir,cpls[0],ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings[0],Nt)
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
        indx2=np.where(np.abs(uaenv[int(uaenv.shape[0]/2):,indx]-np.amax(uaenv[int(uaenv.shape[0]/2):,indx])/np.exp(1.0))<2e-3)
        xmax.append(x[indx])
        dw.append(np.abs(y[int(uaenv.shape[0]/2)+indx2[0][0]]))

    w0a=4.0
    zRa = np.pi*w0a**2
    dwexpctd=[w0a*np.sqrt(2.0*np.pi)*np.sqrt(1+x**2/zRa**2) for x in xmax]
    dwdr=[(x-y)/x for x,y in zip(dw,dwexpctd)]
    
    assert np.mean(np.abs(dwdr))<WIDTH_PRECISION

    en1=np.abs(np.loadtxt(path+'/energy1.txt'))
    assert np.abs((np.amax(en1)-np.amin(en1))/np.amax(en1)<ENERGY_CONSERVATION)

