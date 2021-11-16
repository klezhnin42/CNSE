#!/usr/bin/python3


import numpy as np

def DiffractionHalfStep(va,vga,cvph,kxm,kym,k2xm,k2ym,dt):
    return np.exp(0.25*1j*dt*(k2xm+k2ym) - 0.5*1j*dt*cvph*(kxm*vga[0] + kym*vga[1]))*va

def IntegrationStep(f0,ua,ub,vga,vgb,cvph1,cvph2,kxm,kym,k2xm,k2ym,dt,Es,coupling):
    # taking FFT from initial envelopes
    va=np.fft.fft2(ua) 
    vb=np.fft.fft2(ub)
    
    #first half-step wrt vg d/dx term & diffraction term
    vna1=DiffractionHalfStep(va,vga,cvph1,kxm,kym,k2xm,k2ym,dt)
    una1=np.fft.ifft2(vna1);
    
    vnb1=DiffractionHalfStep(vb,vgb,cvph2,kxm,kym,k2xm,k2ym,dt)
    unb1=np.fft.ifft2(vnb1);
    
    #applying self-focusing term
    pot=Es*((np.abs(una1))**2+(np.abs(unb1))**2);
    una2=np.exp(-1j*dt*pot)*una1;
    unb2=np.exp(-1j*dt*pot)*unb1;
    
    #handling source term due to density perturbations
    A11 = 1.0-(dt*np.abs(f0))**2/2.0; # sqrt(1.-(dt.*abs(f0)).^2) ;
    A12 = f0*dt;
    A21 = -1.0*np.conjugate(f0)*dt;
    A22 = 1.0-(dt*np.abs(f0))**2/2.0; # sqrt(1.-(dt.*abs(f0)).^2) ;   

    a = (A11*una2+A12*unb2);
    b = (A21*una2+A22*unb2);

    f = f0+dt*a*np.conjugate(b)*coupling

    una2 = a; 
    unb2 = b; 
    f0 = f; 
    
    #final half-step wrt vg d/dx term & diffraction term
    vna2=np.fft.fft2(una2);
    va=DiffractionHalfStep(vna2,vga,cvph1,kxm,kym,k2xm,k2ym,dt)
    
    vnb2=np.fft.fft2(unb2);
    vb=DiffractionHalfStep(vnb2,vgb,cvph2,kxm,kym,k2xm,k2ym,dt)

    ua=np.fft.ifft2(va);
    ub=np.fft.ifft2(vb);
    return ua,ub,f0
