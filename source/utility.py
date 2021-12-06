#!/usr/bin/python3

import numpy as np


# function to define a Gaussian envelope; currently may only include 1 value of group velocity

def Gaussian_envelope(angleb,ampb,durb,w0b,xbini,ybini,dist_to_focus,xx,yy):
    zRb = np.pi*w0b**2 # Rayleigh length
    qbini = -dist_to_focus+1j*zRb # Complex parameter of the beam
    # rotating coordinate system to correctly define a beam
    xxb = (xx-xbini)*np.cos(angleb/180*np.pi)-(yy-ybini)*np.sin(angleb/180*np.pi)
    yyb = (xx-xbini)*np.sin(angleb/180*np.pi)+(yy-ybini)*np.cos(angleb/180*np.pi)
    return ampb*np.exp(-1j*(yyb**2)/(2*qbini))*np.exp(-(xxb)**2/(durb**2))

# function to find the optimal pump origin for efficient seed-pump interaction; assumes seed propagation along x with y=0

def Pump_origin(angleb, vgr1, vgr2, xint, xs0, dura):
    #first, let's calculate a distance from pump origin to 'interaction location' (xint,0)
    rb = vgr2/vgr1*(np.abs(xint-xs0)+0.5*dura)
    # now, let's calulate the pump origin coordinates
    xp0 = xint - rb*np.cos(angleb/180*np.pi)
    yp0 = rb*np.sin(angleb/180*np.pi)
    return xp0, yp0


# function to measure beam width and duration, has some flaws for beams intersecting boundaries

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
