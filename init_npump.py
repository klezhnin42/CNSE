#!/usr/bin/python3

#technical parameters for the solver
import numpy as np
import itertools 
from source.utility import *

Lx = 1600/(2*np.pi); # period 2*pi*L, normalized to 2pi \lambda
Ly = 1600/(2*np.pi); # period 2*pi*L, normalized to 2pi \lambda
Nx = 2*128 # number of harmonics
Ny = 2*128 # number of harmonics
dx = Lx/Nx;
tfinal = 1200; # final time, in omega_1^-1
dt = 1*dx;# tfinal/Nt; # time step
Nt = tfinal/dt; # number of time slices

#solver features
x = (2*np.pi/Nx)*np.linspace(-int(Nx/2),int(Nx/2 -1),Nx)*Lx; # x coordinate
kx = np.concatenate([np.linspace(0,Nx/2-1,int(Nx/2-1)+1),[0],np.linspace(-Nx/2+1,-1,int(Nx/2-1))])/Lx; # wave vector
y = (2*np.pi/Ny)*np.linspace(-int(Ny/2),int(Ny/2 -1),Ny)*Ly; # y coordinate
ky = np.concatenate([np.linspace(0,Ny/2-1,int(Ny/2-1)+1),[0],np.linspace(-Ny/2+1,-1,int(Ny/2-1))])/Ly; # wave vector
[xx,yy]=np.meshgrid(x,y);
[k2xm,k2ym]=np.meshgrid(kx**2,ky**2);
[kxm,kym]=np.meshgrid(kx,ky);

maindir='./';

# initial conditions of the laser & constants are defined

#laser group velocity direction
anglea = 0;
#angleb = 60
vga  = [np.cos(anglea/180*np.pi), -np.sin(anglea/180*np.pi)];  # group velocity of pulses, in c
#vgb  = [np.cos(angleb/180*np.pi), -np.sin(angleb/180*np.pi)]; # group velocity of pulses, in c

#coupling constants calculation
#theta=np.abs(anglea-angleb);  #oblique angle wrt x axis
wpw1=0.2; # plasma omega to w1
w1w2=1;   # ratio of frequencies
Vfrs = wpw1**2/4; # coupling const in envelope eqns
#Wfrs = wpw1*(1-np.cos(theta/180*np.pi))*(1-wpw1**2); # coupling const in density eqn
Es = 0.0; # 3/16*wpw1^2;
cvph1=np.sqrt(1-wpw1**2);
cvph2=np.sqrt(1-wpw1**2/(1+wpw1)**2);
cvph3=np.sqrt(1-wpw1**2/(1+wpw1)**2)
k1=np.sqrt(1-wpw1**2)
k2=np.sqrt((1+wpw1)**2-wpw1**2)
amp=0.05
w2w1=1+wpw1

#definition of two laser envelopes

#seed conditions
dura = 50;
w0a = 4; 
zRa = np.pi*w0a**2 ;              # Rayleigh range
xaini = -600 ;         # initial position, with respect to focus
yaini = 0 ;
yafocus = 1200 ;      # distance to focus transversly, does not matter here
qaini = -yafocus+1j*zRa; # Complex parameter of the beam

xxa = (xx-xaini)*np.cos(anglea/180*np.pi)-(yy-yaini)*np.sin(anglea/180*np.pi);
yya = (xx-xaini)*np.sin(anglea/180*np.pi)+(yy-yaini)*np.cos(anglea/180*np.pi);

qaini = qaini+xxa

ua0 = amp*np.exp(-1j*(yya)**2/(2*qaini))*np.exp(-(xxa)**2/(dura**2))

ua0energy=np.abs(sum(sum(ua0*np.conjugate(ua0))))

ua0 = ua0 * 0.1

# pump conditions
ub0s=[]
uc0s=[]
vgbs=[]
vgcs=[]
couplings1=[]
couplings2=[]

durb = 50;
w0b = 20;
rbini = np.linspace(329,329,1);
angleb = np.linspace(60,90,1);

cpls = list(itertools.product(rbini, angleb))

#list of interaction locations, x coordinates only, y=0 for all
xintlist = np.linspace(-200,0,2)
#xintlist = np.linspace(-350,200,6)
xintlist = np.concatenate((xintlist,xintlist))
angles= np.linspace(60,60,2)
angles=np.concatenate((angles,-1*angles))

#mask=[1,-1,1,-1,1,-1]
#angles=[x*y for x,y in zip(angles,mask)]
#angles=np.array(angles)

couplings=[]
pumps=[]
vgbs=[]

for idx in range(len(angles)):   
    vgb  = [np.cos(angles[idx]/180*np.pi), -np.sin(angles[idx]/180*np.pi)] # group velocity of pulses, in c
    vgbs.append(vgb)
    theta=np.abs(anglea-angles[idx])  #oblique angle wrt x axis
    couplings.append(Vfrs*1.0/wpw1*(k1**2+k2**2-2.0*k1*k2*np.cos(theta/180*np.pi))) # coupling const in density eqn

    ybfocus = 0.0

    # get optimal pump location for interaction at (xint, 0.0)
    xbini, ybini = Pump_origin(angles[idx], cvph1, cvph2, xintlist[idx], xaini, dura)

    # defining a beam envelope with unit max amplitude
    pumps.append(Gaussian_envelope(angles[idx],1.0,durb,w0b,xbini,ybini,ybfocus,xx,yy))
  

# calculating max amplitude so the pump and seed have the same energy

alphaen=0

for unx in pumps:
    alphaen=alphaen+np.abs(sum(sum(unx*np.conjugate(unx))))

ampb = np.sqrt(ua0energy/alphaen)

# defining an actual pump envelope

for i in range(len(pumps)):
    pumps[i]=pumps[i]*ampb

uvec=[ua0]
for li in pumps:
    uvec.append(li)

vgvec=[vga]
cvvec=[cvph1]

for vgi in vgbs:
    vgvec.append(vgi)
    cvvec.append(cvph2)

#mask=[0,0,1,0,0,0]

#couplings=[x*y for x,y in zip(couplings,mask)]
#print(couplings)

datainputs=[[maindir,cpls[0],uvec,vgvec,cvvec,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt]]
