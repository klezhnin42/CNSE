#!/usr/bin/python3

import solver
import multiprocessing
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import date
from datetime import datetime
from numpy import save
from numpy import load

# driver
def Simulation(maindir,ua,ub,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling,Nt):
    #create a random directory within maindir
    now = datetime.now()
    timestamp = now.strftime("%d%m%y_%H%M%S")    
    path=os.path.join(maindir,timestamp)
    path=path+'_'+str(multiprocessing.current_process())
    os.mkdir(path)
    
    # taking initial density perturbation to be zero
    f0 = np.zeros(ua.shape); 

    #initial output
    u = np.abs(ua+ub); # envelope to plot
    uaang=np.angle(ua)
    ubang=np.angle(ub)
    save(path+'/data'+str(0)+'.npy',ua,ub)
    #for in1 in range(len(x)):
    #    for in2 in range(len(y)):
    #        if np.abs(ua[in1,in2])<1e-2*np.amax(np.abs(ua)):
    #            uaang[in1,in2]=None
    #        if np.abs(ub[in1,in2])<1e-2*np.amax(np.abs(ub)):
    #            ubang[in1,in2]=None
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    pos=ax1.imshow(u,vmin=0.0,vmax=0.5,extent =[x.min(), x.max(), y.min(), y.max()]) # plot IC at Nstep=i
    fig.colorbar(pos, ax=ax1, orientation="horizontal")
    pos2=ax2.imshow(uaang,cmap='RdBu',extent =[x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(pos2, ax=ax2,orientation="horizontal")
    pos3=ax3.imshow(ubang,cmap='RdBu',extent =[x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(pos3, ax=ax3,orientation="horizontal")
    fig.suptitle('t='+str(int(0*dt)))
    plt.savefig(path+'/EM_'+str(0)+'.png')
    plt.close()

    #parameters to collect
    energy1=[]
    energy2=[]

    for i in range(int(Nt)):
        ua,ub,f0=solver.IntegrationStep(f0,ua,ub,vga,vgb,cvph1,cvph2,kxm,kym,k2xm,k2ym,dt,Es,coupling)
        if(i%10==0):
            #collecting parameters
            energy1.append(sum(sum(np.abs(ua)**2)));
            energy2.append(sum(sum(np.abs(ub)**2)));
            
            #image output
            u = np.abs(ua+ub); # envelope to plot
            uaang=np.angle(ua)
            ubang=np.angle(ub)
            save(path+'/data_ua_'+str(i)+'.npy',ua)
            save(path+'/data_ub_'+str(i)+'.npy',ub)
            #if (i%100==0):
            #    for in1 in range(len(x)):
            #        for in2 in range(len(y)):
            #            if np.abs(ua[in1,in2])<1e-2*np.amax(np.abs(ua)):
            #                uaang[in1,in2]=None
            #            if np.abs(ub[in1,in2])<1e-2*np.amax(np.abs(ub)):
            #                ubang[in1,in2]=None
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.set_figheight(8)
            fig.set_figwidth(16)
            pos=ax1.imshow(u,vmin=0.0,vmax=0.5,extent =[x.min(), x.max(), y.min(), y.max()]) # plot IC at Nstep=i
            fig.colorbar(pos, ax=ax1, orientation="horizontal")
            pos2=ax2.imshow(uaang,vmin=-np.pi,vmax=np.pi, cmap='RdBu',extent =[x.min(), x.max(), y.min(), y.max()])
            fig.colorbar(pos2, ax=ax2,orientation="horizontal")
            pos3=ax3.imshow(ubang,vmin=-np.pi,vmax=np.pi, cmap='RdBu',extent =[x.min(), x.max(), y.min(), y.max()])
            fig.colorbar(pos3, ax=ax3,orientation="horizontal")
            fig.suptitle('t='+str(int(i*dt)))
            plt.savefig(path+'/EM_'+str(i)+'.png')
            plt.close()
    np.savetxt(path+'/energy1.txt',energy1)
    np.savetxt(path+'/energy2.txt',energy2)
    return path
