#!/usr/bin/python3

#technical parameters for the solver
import numpy as np
import itertools 
from source.utility import *

Lx = 800/(2*np.pi); # period 2*pi*L, normalized to 2pi \lambda
Ly = 100/(2*np.pi); # period 2*pi*L, normalized to 2pi \lambda
Nx = 2*128 # number of harmonics
Ny = 2*8 # number of harmonics
dx = Lx/Nx;
tfinal = 500; # final time, in omega_1^-1
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

k2xm=np.zeros((Ny,Nx))
k2ym=np.zeros((Ny,Nx))

maindir='./';

# initial conditions of the laser & constants are defined

#laser group velocity direction
anglea = 0
vga  = [np.cos(anglea/180*np.pi), -np.sin(anglea/180*np.pi)];  # group velocity of pulses, in c

#coupling constants calculation

wpw1=0.2 # plasma omega to w1
Vfrs = wpw1**2/4 # coupling const in envelope eqns
Es = 0.0 # 3/16*wpw1^2;
cvph1=1.0 #np.sqrt(1-wpw1**2);
cvph2=1.0 #np.sqrt(1-wpw1**2/(1+wpw1)**2);
k1=np.sqrt(1-wpw1**2)
k2=np.sqrt((1+wpw1)**2-wpw1**2)
amp=0.01
w2w1=1+wpw1

#definition of two laser envelopes

#seed conditions
dura = 50
w0a = 100
Nrefra = np.sqrt(1-wpw1**2) 
zRa = np.pi*w0a**2*2.0*np.pi*Nrefra              # Rayleigh range
xaini = -200 ;         # initial position, with respect to focus
yaini = 0 ;
yafocus = 0 ;      # distance to focus transversly, does not matter here
qaini = -yafocus+1j*zRa; # Complex parameter of the beam

xxa = (xx-xaini)*np.cos(anglea/180*np.pi)-(yy-yaini)*np.sin(anglea/180*np.pi);
yya = (xx-xaini)*np.sin(anglea/180*np.pi)+(yy-yaini)*np.cos(anglea/180*np.pi);

qaini = qaini+xxa

ua0 = amp*qaini/qaini*np.exp(-(xxa)**10/(dura**10))


ua0energy=np.abs(sum(sum(ua0*np.conjugate(ua0))))


# pump conditions

durb = 200;
w0b = 100;
Nrefrb = np.sqrt(1-wpw1**2/(w2w1)**2)
kb = w2w1*Nrefrb
zRb = np.pi*w0b**2*2*np.pi*Nrefrb*w2w1
rbini = np.linspace(100,329,1);
angleb = np.linspace(0,90,1);

cpls = list(itertools.product(rbini, angleb))

couplings=[]
pumps=[]
vgbs=[]

for idx in range(len(angleb)):   
    vgb  = [np.cos(angleb[idx]/180*np.pi), -np.sin(angleb[idx]/180*np.pi)] # group velocity of pulses, in c
    vgbs.append(vgb)
    theta=np.abs(anglea-angleb[idx])  #oblique angle wrt x axis
    couplings.append(Vfrs*1.0/wpw1*(k1**2+k2**2-2.0*k1*k2*np.cos(theta/180*np.pi))) # coupling const in density eqn

    xbfocus = 0.0

    xbini, ybini = -200, 0 

    qbini = -xbfocus+1j*zRb

    xxb = (xx-xbini)*np.cos(angleb[idx]/180*np.pi)-(yy-ybini)*np.sin(angleb[idx]/180*np.pi)
    yyb = (xx-xbini)*np.sin(angleb[idx]/180*np.pi)+(yy-ybini)*np.cos(angleb[idx]/180*np.pi)
    qbini = qbini+xxb

    # defining a beam envelope with unit max amplitude
    pumps.append(qbini/qbini)
  

# calculating max amplitude so the pump and seed have the same energy

alphaen=np.sum(pumps[0])

ampb = 10*amp

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

datainputs=[[maindir,cpls[0],uvec,vgvec,cvvec,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings,Nt]]
