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

FRS_PRECISION=2e-2
ENERGY_CONSERVATION=1e-6

# Functions to integrate FRS solution

from scipy.integrate import quad
from scipy.special import iv
from numpy import heaviside
import scipy

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x,a0,width,gamma2c,z,t):
        return np.real(func(x,a0,width,gamma2c,z,t))
    def imag_func(x,a0,width,gamma2c,z,t):
        return np.imag(func(x,a0,width,gamma2c,z,t))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def ksi(gamma2c,x,z,t):
    return 2.0*gamma2c*np.sqrt((z - x)*(t - z + x)+0j)

def integrand(x,a0,width,gamma2c,z,t):
    return gamma2c*np.sqrt((z-x)/(t-z+x)+0j)*iv(1,ksi(gamma2c,x,z,t)+0j)*heaviside(z-x,0.5)*heaviside(t-z+x,0.5)*a0*np.exp(-x**10/width**10)

def analytical_envelope(a0,width,gamma2c,z,t):
    intgrnd=complex_quadrature(integrand,-np.inf,np.inf,args=(a0,width,gamma2c,z,t))
    return intgrnd


def test_frs0():
    from .init_frs0 import maindir,cpls,ua0,ub0,vga,vgb,cvph1,cvph2,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.Simulation(maindir,cpls[0],ua0,ub0,vga,vgb,cvph1,cvph2,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings[0],Nt)
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))

    # array to calculate mean envelope error    
    derr=[]

    # parameters to calculate the analytical solution
    wpw1=0.2
    k1=np.sqrt(1-wpw1**2)
    k2=np.sqrt((1+wpw1)**2-wpw1**2)
    width=50
    apump=0.1
    V1=wpw1**2/4
    W1=1.0/wpw1*(k1**2+k2**2-2.0*k1*k2*np.cos(0.0/180*np.pi))
    VFRS=np.sqrt(V1*W1)
    gamma2c=apump*VFRS
    a0=0.01
    times=np.linspace(0,500,11)

    i=0
    # loop to collect envelope integration errors
    for file in fllst[::10]:
        ua=np.load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        uaenv1d=uaenv[8,:] 
        zz=np.linspace(-400,400,256)
        envlp=[np.real(analytical_envelope(a0,width,gamma2c,z+200,times[i])[0]+a0*np.exp(-(z-times[i]+200)**10/width**10)) for z in zz]
        derr.append([(x-y)/a0 for x,y in zip(envlp,uaenv1d)]) 
        i=i+1 

    # checking whether maximum mean error over time is small enough
    assert np.amax([np.mean(np.abs(x)) for x in derr]) < FRS_PRECISION

    # check energy conservation
    en1=np.abs(np.loadtxt(path+'/energy1.txt'))
    en2=np.abs(np.loadtxt(path+'/energy2.txt'))
    entot=en1+en2
    assert np.abs((np.amax(entot)-np.amin(entot))/np.amin(entot)<ENERGY_CONSERVATION)




#testing the same FRS conditions as before but for Multi-Pump solver
def test_frs1():
    from .init_frs1 import maindir,cpls,uvec,vgvec,cvvec,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.SimulationMultiPumpMultiBeat(maindir,cpls,uvec,vgvec,cvvec,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt)
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))

    # array to calculate mean envelope error
    derr=[]

    # parameters to calculate the analytical solution
    wpw1=0.2
    k1=np.sqrt(1-wpw1**2)
    k2=np.sqrt((1+wpw1)**2-wpw1**2)
    width=50
    apump=0.1
    V1=wpw1**2/4
    W1=1.0/wpw1*(k1**2+k2**2-2.0*k1*k2*np.cos(0.0/180*np.pi))
    VFRS=np.sqrt(V1*W1)
    gamma2c=apump*VFRS
    a0=0.01
    times=np.linspace(0,500,11)

    i=0
    # loop to collect envelope integration errors
    for file in fllst[::10]:
        ua=np.load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        uaenv1d=uaenv[8,:]
        zz=np.linspace(-400,400,256)
        envlp=[np.real(analytical_envelope(a0,width,gamma2c,z+200,times[i])[0]+a0*np.exp(-(z-times[i]+200)**10/width**10)) for z in zz]
        derr.append([(x-y)/a0 for x,y in zip(envlp,uaenv1d)])
        i=i+1

    # checking whether maximum mean error over time is small enough
    assert np.amax([np.mean(np.abs(x)) for x in derr]) < FRS_PRECISION

    # check energy conservation
    en=np.abs(np.loadtxt(path+'/energy.txt'))
    entot=[x[0]+x[1] for x in en]
    assert np.abs((np.amax(entot)-np.amin(entot))/np.amin(entot)<ENERGY_CONSERVATION)



#testing the Multi-Pump solver for two pump envelopes
def test_frs2():
    from .init_frs2 import maindir,cpls,uvec,vgvec,cvvec,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.SimulationMultiPump(maindir,cpls,uvec,vgvec,cvvec,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt)
    fllst=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("data_ua"):
                fllst.append(file)
    fllst.sort(key=lambda f: int(re.sub(r'\D', '', f)))

    # array to calculate mean envelope error
    derr=[]

    # parameters to calculate the analytical solution
    wpw1=0.2
    k1=np.sqrt(1-wpw1**2)
    k2=np.sqrt((1+wpw1)**2-wpw1**2)
    width=50
    apump=0.1
    V1=wpw1**2/4
    W1=1.0/wpw1*(k1**2+k2**2-2.0*k1*k2*np.cos(0.0/180*np.pi))
    VFRS=np.sqrt(V1*W1)
    gamma2c=apump*VFRS
    a0=0.01
    times=np.linspace(0,500,11)

    i=0
    # loop to collect envelope integration errors
    for file in fllst[::10]:
        ua=np.load(path+'/'+file)
        uaenv=np.sqrt(np.abs(ua*np.conjugate(ua)))
        uaenv1d=uaenv[8,:]
        zz=np.linspace(-400,400,256)
        envlp=[np.real(analytical_envelope(a0,width,gamma2c,z+200,times[i])[0]+a0*np.exp(-(z-times[i]+200)**10/width**10)) for z in zz]
        derr.append([(x-y)/a0 for x,y in zip(envlp,uaenv1d)])
        i=i+1

    # checking whether maximum mean error over time is small enough
    assert np.amax([np.mean(np.abs(x)) for x in derr]) < FRS_PRECISION

    # check energy conservation
    en=np.abs(np.loadtxt(path+'/energy.txt'))
    entot=[x[0]+x[1]+x[2] for x in en]
    assert np.abs((np.amax(entot)-np.amin(entot))/np.amin(entot)<ENERGY_CONSERVATION)
