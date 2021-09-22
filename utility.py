#!/usr/bin/python3

import numpy as np

def waist_duration(ua,vga):
    #finding beam center of mass
    arry=np.abs(ua)
    iy,ix=np.where(arry==np.amax(arry))
    ix=ix[0]
    iy=iy[0]
    xini=x[ix]
    yini=y[iy]
    #recovering propagation angle
    phi=np.arcsin(vga[1]/np.sqrt(vga[0]**2+vga[1]**2))/np.pi*180
    #finding locations where it is half-maximum
    yl,xl=np.where(np.abs(arry-np.amax(arry)/2.0)<1e-3)
    yl=-yl
    #shifting coordinates to use convenient axes
    xarr=np.cos(phi/180*np.pi)*(x[xl]-xini)-np.sin(phi/180*np.pi)*(y[yl]-yini)
    yarr=np.cos(phi/180*np.pi)*(y[yl]-yini)+np.sin(phi/180*np.pi)*(x[xl]-xini)
    #return waist and duration
    return np.abs(np.amax(yarr)-np.amin(yarr)),np.abs(np.amax(xarr)-np.amin(xarr))
