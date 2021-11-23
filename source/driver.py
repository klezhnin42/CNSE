#!/usr/bin/python3

from source import solver
from source import output
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import date
from datetime import datetime
from numpy import save
from numpy import load

# driver
def Simulation(maindir,params,ua,ub,vga,vgb,cvph1,cvph2,w2w0,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling,Nt):
    try:
        rb,phib=params[0],params[1]
        foldname='rb_'+str(rb)+'_phib_'+str(phib)
        path=os.path.join(maindir,foldname)
        os.mkdir(path)
    except:
        now = datetime.now()
        timestamp = now.strftime("%d%m%y_%H%M%S")
        foldname = timestamp + str(multiprocessing.current_process())
        path=os.path.join(maindir,foldname)
        os.mkdir(path)    
    # taking initial density perturbation to be zero
    f0 = np.zeros(ua.shape); 

    #parameters to collect
    energy1=[]
    energy2=[]

    for i in range(int(Nt)):
        if(i%10==0):
            #collecting parameters
            energy1.append(np.abs(sum(sum(ua*np.conjugate(ua)))))
            energy2.append(np.abs(sum(sum(ub*np.conjugate(ub)))))

            #plotting basic information & dumping envelopes
            output.basic_output(path,ua,ub,x,y,i,dt)
        # integration timestep
        ua,ub,f0=solver.IntegrationStep(f0,ua,ub,vga,vgb,cvph1,cvph2,w2w0,kxm,kym,k2xm,k2ym,dt,Es,coupling)
    np.savetxt(path+'/energy1.txt',energy1)
    np.savetxt(path+'/energy2.txt',energy2)
    return path
