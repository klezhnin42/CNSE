#!/usr/bin/python3

import numpy as np
from numpy import save
from matplotlib import pyplot as plt

def basic_output(path,ua,ub,x,y,i,dt):
    # save envelopes
    save(path+'/data_ua_'+str(i)+'.npy',ua)
    save(path+'/data_ub_'+str(i)+'.npy',ub)

    #image output
    u = np.abs(ua+ub); # envelope to plot
    uaang=np.angle(ua)
    ubang=np.angle(ub)
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
