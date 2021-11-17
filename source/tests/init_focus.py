#!/usr/bin/python3

#technical parameters for the solver
import numpy as np
from numpy import save
import itertools 

Lx = 800/(2*np.pi); # period 2*pi*L, normalized to 2pi \lambda
Ly = 800/(2*np.pi); # period 2*pi*L, normalized to 2pi \lambda
Nx = 2*256; # number of harmonics
Ny = 2*256; # number of harmonics
dx = Lx/Nx;
tfinal = 400; # final time, in omega_1^-1
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
vga  = [np.cos(anglea/180*np.pi), -np.sin(anglea/180*np.pi)];  # group velocity of pulses, in c

#coupling constants calculation

wpw1=0.002; # plasma omega to w1
Vfrs = wpw1**2/4; # coupling const in envelope eqns
Es = 0.0; # 3/16*wpw1^2;
cvph1=np.sqrt(1-wpw1**2);
cvph2=np.sqrt(1-wpw1**2/(1+wpw1)**2);
k1=np.sqrt(1-wpw1**2)
k2=np.sqrt((1+wpw1)**2-wpw1**2)
amp=0.1


#definition of two laser envelopes

#seed conditions
dura = 50;
w0a = 4; 
zRa = np.pi*w0a**2 ;              # Rayleigh range
xaini = -200 ;         # initial position, with respect to focus
yaini = 0 ;
xafocus = 200 ;      # distance to focus transversly
qaini = -xafocus+1j*zRa; # Complex parameter of the beam

xxa = (xx-xaini)*np.cos(anglea/180*np.pi)-(yy-yaini)*np.sin(anglea/180*np.pi);
yya = (xx-xaini)*np.sin(anglea/180*np.pi)+(yy-yaini)*np.cos(anglea/180*np.pi);

qaini = qaini+xxa

ua0 = amp*np.exp(-1j*(yya)**2/(2*qaini))*np.exp(-(xxa)**2/(dura**2))

ua0energy=abs(sum(sum(ua0*np.conjugate(ua0))))

# pump conditions; just keep it to keep the main part of the code intact for now; later when we add an opportunity to add a custom number of envelopes it will be fixed
ub0s=[]
vgbs=[]
couplings=[]

durb = 50;
w0b = 10;
zRb = np.pi*w0b**2 ;              # Rayleigh range
rbini = np.linspace(100,350,1);
angleb = np.linspace(0,90,1);

cpls = list(itertools.product(rbini, angleb))

for rb,phib in cpls:   
    vgb  = [np.cos(phib/180*np.pi), -np.sin(phib/180*np.pi)]; # group velocity of pulses, in c
    theta=np.abs(anglea-phib);  #oblique angle wrt x axis
    Wfrs = 0.0* 1.0/wpw1*(k1**2+k2**2-2.0*k1*k2*np.cos(theta/180*np.pi)) # coupling const in density eqn
    xbini = -rb*np.cos(phib/180*np.pi) ;         # initial position, with respect to focus
    ybini = rb*np.sin(phib/180*np.pi)  ;
    xbfocus = 200;        # distance to focus transversly
    qbini = -xbfocus+1j*zRb ; # Complex parameter of the beam
    xxb = (xx-xbini)*np.cos(phib/180*np.pi)-(yy-ybini)*np.sin(phib/180*np.pi);
    yyb = (xx-xbini)*np.sin(phib/180*np.pi)+(yy-ybini)*np.cos(phib/180*np.pi);
    alpha=np.exp(-1j*(yyb**2)/(2*qbini))*np.exp(-(xxb)**2/(durb**2))
    alphasum = abs(sum(sum(alpha*np.conjugate(alpha))))
    ampb = np.sqrt(ua0energy/alphasum)
    ub0 = 0.0*ampb*np.exp(-1j*(yyb**2)/(2*qbini))*np.exp(-(xxb)**2/(durb**2))
    ub0s.append(ub0)
    vgbs.append(vgb)
    couplings.append(Vfrs*Wfrs*0.0)

datainputs=[[maindir,cpl,ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling,Nt] for cpl,ub0,vgb,coupling in zip(cpls,ub0s,vgbs,couplings)]
