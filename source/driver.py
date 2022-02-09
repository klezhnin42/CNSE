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

#driver for two-pump simulation
def SimulationTwoPump(maindir,params,ua,ub,uc,vga,vgb,vgc,cvph1,cvph2,cvph3,w2w0,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling1,coupling2,Nt):
    try:
        pr1,pr2=params[0],params[1]
        foldname='pr1_'+str(pr1)+'_pr2_'+str(pr2)
        path=os.path.join(maindir,foldname)
        os.mkdir(path)
    except:
        now = datetime.now()
        timestamp = now.strftime("%d%m%y_%H%M%S")
        foldname = timestamp + str(multiprocessing.current_process())
        path=os.path.join(maindir,foldname)
        os.mkdir(path)
    # taking initial density perturbation to be zero
    f0 = np.zeros(ua.shape)
    g0 = np.zeros(ua.shape) 
 
    #parameters to collect
    energy1=[]
    energy2=[]
    energy3=[]

    for i in range(int(Nt)):
        if(i%10==0):
            #collecting parameters
            energy1.append(np.abs(sum(sum(ua*np.conjugate(ua)))))
            energy2.append(np.abs(sum(sum(ub*np.conjugate(ub)))))
            energy3.append(np.abs(sum(sum(uc*np.conjugate(uc)))))
            #plotting basic information & dumping envelopes
            output.basic_output(path,ua,ub+uc,x,y,i,dt)
        # integration timestep
        ua,ub,uc,f0,g0=solver.IntegrationStepTwoPump(f0,g0,ua,ub,uc,vga,vgb,vgc,cvph1,cvph2,cvph3,w2w0,kxm,kym,k2xm,k2ym,dt,Es,coupling1,coupling2)
    np.savetxt(path+'/energy1.txt',energy1)
    np.savetxt(path+'/energy2.txt',energy2)
    np.savetxt(path+'/energy3.txt',energy3)
    return path


#driver for two-pump simulation with a unified beating
def SimulationTwoPumpOneBeat(maindir,params,ua,ub,uc,vga,vgb,vgc,cvph1,cvph2,cvph3,w2w0,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling1,coupling2,Nt):
    try:
        pr1,pr2=params[0],params[1]
        foldname='pr1_'+str(pr1)+'_pr2_'+str(pr2)
        path=os.path.join(maindir,foldname)
        os.mkdir(path)
    except:
        now = datetime.now()
        timestamp = now.strftime("%d%m%y_%H%M%S")
        foldname = timestamp + str(multiprocessing.current_process())
        path=os.path.join(maindir,foldname)
        os.mkdir(path)
    # taking initial density perturbation to be zero
    f0 = np.zeros(ua.shape)

    #parameters to collect
    energy1=[]
    energy2=[]
    energy3=[]

    for i in range(int(Nt)):
        if(i%10==0):
            #collecting parameters
            energy1.append(np.abs(sum(sum(ua*np.conjugate(ua)))))
            energy2.append(np.abs(sum(sum(ub*np.conjugate(ub)))))
            energy3.append(np.abs(sum(sum(uc*np.conjugate(uc)))))
            #plotting basic information & dumping envelopes
            output.debug_output(path,ua,[ub,uc],f0,i)
        # integration timestep
        ua,ub,uc,f0=solver.IntegrationStepTwoPumpOneBeat(f0,ua,ub,uc,vga,vgb,vgc,cvph1,cvph2,cvph3,w2w0,kxm,kym,k2xm,k2ym,dt,Es,coupling1,coupling2)
    np.savetxt(path+'/energy1.txt',energy1)
    np.savetxt(path+'/energy2.txt',energy2)
    np.savetxt(path+'/energy3.txt',energy3)
    return path



#driver for multi-pump simulation
def SimulationMultiPump(maindir,params,uvec,vgvec,cvvec,w2w0,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt):
    try:
        pr1,pr2=params[0],params[1]
        foldname='pr1_'+str(pr1)+'_pr2_'+str(pr2)
        path=os.path.join(maindir,foldname)
        os.mkdir(path)
    except:
        now = datetime.now()
        timestamp = now.strftime("%d%m%y_%H%M%S")
        foldname = timestamp + str(multiprocessing.current_process())
        path=os.path.join(maindir,foldname)
        os.mkdir(path)
    # taking initial density perturbation to be zero
    f0 = np.zeros(uvec[0].shape)
    #f0s = []
    #for i in range(len(uvec)-1):
    #    f0s.append(f0)
    #parameters to collect
    energys=[]

    for jj in range(int(Nt)):
        if(jj%10==0):
            #collecting parameters
            lst_en=[]
            for dsf in uvec:
                lst_en.append(np.abs(np.sum(dsf*np.conjugate(dsf))))
            energys.append(lst_en)
            #plotting basic information & dumping envelopes
            output.debug_output(path,uvec[0],uvec[1:],f0,jj)
        # integration timestep
        uvec,f0=solver.IntegrationStepMultiPump(f0,uvec,vgvec,cvvec,w2w0,kxm,kym,k2xm,k2ym,dt,Es,couplings)
    np.savetxt(path+'/energy.txt',energys)
    return path
