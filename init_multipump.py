#!/usr/bin/python3

#technical parameters for the solver
import numpy as np
import itertools 
from source.utility import *

Lx = 1600/(2*np.pi); # period 2*pi*L, normalized to 2pi \lambda
Ly = 1600/(2*np.pi); # period 2*pi*L, normalized to 2pi \lambda
Nx = 2*128; # number of harmonics
Ny = 2*128; # number of harmonics
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

# pump conditions
ub0s=[]
vgbs=[]
couplings=[]

durb = 50;
w0b = 20;
rbini = np.linspace(329,329,1);
angleb = np.linspace(60,90,1);

cpls = list(itertools.product(rbini, angleb))

#list of interaction locations, x coordinates only, y=0 for all
xintlist = [-400,-300,-200,-100,0.0,100,200]  

for rb,phib in cpls:   
    vgb  = [np.cos(phib/180*np.pi), -np.sin(phib/180*np.pi)]; # group velocity of pulses, in c
    theta=np.abs(anglea-phib);  #oblique angle wrt x axis
    Wfrs = 1.0/wpw1*(k1**2+k2**2-2.0*k1*k2*np.cos(theta/180*np.pi)) # coupling const in density eqn

    ybfocus = 0.0

    # define variable to collect all envelopes to for further normalization

    alpha = 0.0 * ua0

    for xint in xintlist:
        # get optimal pump location for interaction at (xint, 0.0)
        xbini, ybini = Pump_origin(phib, cvph1, cvph2, xint, xaini, dura)

        # defining a beam envelope with unit max amplitude
        alpha = alpha + Gaussian_envelope(phib,1.0,durb,w0b,xbini,ybini,ybfocus,xx,yy)

    # calculating max amplitude so the pump and seed have the same energy

    alphasum = np.abs(sum(sum(alpha*np.conjugate(alpha))))
    ampb = np.sqrt(ua0energy/alphasum)

    # defining an actual pump envelope
    ub0 = 0.0*ampb*alpha
    ub0s.append(ub0)
    vgbs.append(vgb)
    couplings.append(Vfrs*Wfrs)

datainputs=[[maindir,cpl,ua0,ub0,vga,vgb,cvph1,cvph2,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling,Nt] for cpl,ub0,vgb,coupling in zip(cpls,ub0s,vgbs,couplings)]
