#!/usr/bin/python3

#technical parameters for the solver
import numpy as np
import itertools 

Lx = 600/(2*np.pi); # period 2*pi*L, normalized to 2pi \lambda
Ly = 600/(2*np.pi); # period 2*pi*L, normalized to 2pi \lambda
Nx = 2*64; # number of harmonics
Ny = 2*64; # number of harmonics
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

maindir='./';

# initial conditions of the laser & constants are defined

#laser group velocity direction
anglea = 0;
angleb = 60
vga  = [np.cos(anglea/180*np.pi), -np.sin(anglea/180*np.pi)];  # group velocity of pulses, in c

#coupling constants calculation
theta=np.abs(anglea-angleb);  #oblique angle wrt x axis
wpw1=0.2; # plasma omega to w1
w1w2=1;   # ratio of frequencies
Vfrs = wpw1**2/4; # coupling const in envelope eqns
Wfrs = wpw1*(1-np.cos(theta/180*np.pi))*(1-wpw1**2); # coupling const in density eqn
coupling=Vfrs*Wfrs
Es = 0.0; # 3/16*wpw1^2;
cvph1=np.sqrt(1-wpw1**2);
cvph2=(1-wpw1)*np.sqrt(1-wpw1**2*(1-wpw1)**2);
amp=0.05
w2w1=1+wpw1

#definition of two laser envelopes

#seed conditions
dura = 50;
w0a = 4; 
zRa = np.pi*w0a**2 ;              # Rayleigh range
xaini = -200 ;         # initial position, with respect to focus
yaini = 0 ;
yafocus = 400 ;      # distance to focus transversly, does not matter here
qaini = -yafocus+1j*zRa; # Complex parameter of the beam

xxa = (xx-xaini)*np.cos(anglea/180*np.pi)-(yy-yaini)*np.sin(anglea/180*np.pi);
yya = (xx-xaini)*np.sin(anglea/180*np.pi)+(yy-yaini)*np.cos(anglea/180*np.pi);

qaini = qaini+xxa

ua0 = amp*np.exp(-1j*(yya)**2/(2*qaini))*np.exp(-(xxa)**2/(dura**2))

# pump conditions
ub0s=[]

#durb = 50;
#w0b = 20;
vgb  = [np.cos(angleb/180*np.pi), -np.sin(angleb/180*np.pi)];
rb = 200;
xbini = -rb*np.cos(angleb/180*np.pi) ;         # initial position, with respect to focus
ybini = rb*np.sin(angleb/180*np.pi)  ;
ybfocus = rb;        # distance to focus transversly
xxb = (xx-xbini)*np.cos(angleb/180*np.pi)-(yy-ybini)*np.sin(angleb/180*np.pi);
yyb = (xx-xbini)*np.sin(angleb/180*np.pi)+(yy-ybini)*np.cos(angleb/180*np.pi);

w0bs = np.linspace(5,300,2)
durbs = np.linspace(5,300,2)


#rbini = np.linspace(50,150,3);
#angleb = np.linspace(10,90,6);

cpls = list(itertools.product(w0bs, durbs))

for w0b,durb in cpls:   
    zRb = np.pi*w0b**2 
    qbini = -ybfocus+1j*zRb ; # Complex parameter of the beam
    ua0en=np.sum(np.abs(ua0*np.conjugate(ua0)))
    alpha = np.exp(-1j*(yyb**2)/(2*qbini))*np.exp(-(xxb)**2/(durb**2))
    alphaen = np.sum(np.abs(alpha*np.conjugate(alpha)))
    ampb = np.sqrt(ua0en/alphaen)
    ub0 = ampb*alpha
    ub0s.append(ub0)

datainputs=[[maindir,cpl,ua0,ub0,vga,vgb,cvph1,cvph2,w2w1,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling,Nt] for cpl,ub0 in zip(cpls,ub0s)]
