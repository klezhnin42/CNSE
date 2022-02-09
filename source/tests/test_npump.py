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

CMP_PRECISION=3e-14
ENERGY_CONSERVATION=3e-3


#testing the same FRS conditions as before but for Multi-Pump solver
def test_npump():

    from .init_npump_tpump_cmp2 import maindir,cpls,ua0,ub0,uc0,vga,vgb,vgc,cvph1,cvph2,cvph3,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings1,couplings2,Nt
    path=driver.SimulationTwoPumpOneBeat(maindir,cpls[0],ua0,ub0,uc0,vga,vgb,vgc,cvph1,cvph2,cvph3,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings1[0],couplings2[0],Nt)


    # check energy conservation
    en1=np.abs(np.loadtxt(path+'/energy1.txt'))
    en2=np.abs(np.loadtxt(path+'/energy2.txt'))
    en3=np.abs(np.loadtxt(path+'/energy3.txt'))
    entot=[x+y+z for x,y,z in zip(en1,en2,en3)]
    assert np.abs((np.amax(entot)-np.amin(entot))/np.amin(entot)<ENERGY_CONSERVATION)


    from .init_npump_tpump_cmp1 import maindir,cpls,uvec,vgvec,cvvec,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt
    path=driver.SimulationMultiPump(maindir,cpls,uvec,vgvec,cvvec,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt)

    # check energy conservation
    en=np.abs(np.loadtxt(path+'/energy.txt'))
    entot=[x[0]+x[1]+x[2] for x in en]
    assert np.abs((np.amax(entot)-np.amin(entot))/np.amin(entot)<ENERGY_CONSERVATION)

    en11=np.array([x[0] for x in en])
    en22=np.array([x[1] for x in en])
    en33=np.array([x[2] for x in en])

    assert np.abs(np.amax(np.abs(en1-en11))/np.amax(en1))<CMP_PRECISION
    assert np.abs(np.amax(np.abs(en2-en22))/np.amax(en2))<CMP_PRECISION
    assert np.abs(np.amax(np.abs(en3-en33))/np.amax(en3))<CMP_PRECISION
