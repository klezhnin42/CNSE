#!/usr/bin/python3

import numpy as np
import os
import sys
from datetime import datetime
from .. import solver
from .. import driver

PRECISION=1e-2

def run_unit_test(maindir,cpl,ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling,Nt):
    path=driver.Simulation(maindir,cpl,ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling,Nt)
    import numpy as np
    from matplotlib import pyplot as plt
    from numpy import load
    import os
    import re
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub('\D', '', f)))
    xmax=[]
    for file in fllst:
        ua=load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        indx=np.argmax(uaenv[int(uaenv.shape[0]/2),:])
        xmax.append(x[indx])
    expcdt=[-200+1*np.sqrt(1-0.05)*10*i*dt for i in range(51)]
    dr=[x-y for x,y in zip(xmax,expcdt)]
    from scipy.signal import savgol_filter
    drhat = savgol_filter(dr, 51, 3)
    if (np.amax(drhat/np.abs(xmax[-1]-xmax[0]))<PRECISION):
        return 0
    else:
        return 1
    
