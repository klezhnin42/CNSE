#!/usr/bin/python3


import numpy as np

#######################################
#half-step for integration of equation: 
# (d/dt + vg * d/dx - ic^2/2 w0 d^2/dx^2) va = 0
# In the adopted normalization (w0 t -> tau, c/w0 d/dx -> d/dx), this equation becomes:
# (d/d tau + vg/c * d/dx - i/2 d^2/dx^2) va = 0
# Note that time is measured in w0^-1, spatial scales - in c/w0, speeds - in c
#######################################

def DiffractionHalfStep(va,vga,cvph,w2w0,kxm,kym,k2xm,k2ym,dt):
    return np.exp(0.25*1j*dt*(k2xm+k2ym)/w2w0 - 0.5*1j*dt*cvph*(kxm*vga[0] + kym*vga[1]))*va

#DiffractionHalfStepPump(vvec[1:],vgvec[1:],cvvec[1:],w2w0,kxm,kym,k2xm,k2ym,dt)
def DiffractionHalfStepPump(vas,vgas,cvphs,w2w0,kxm,kym,k2xm,k2ym,dt):
    for i in range(len(vas)):
        vas[i]=np.exp(0.25*1j*dt*(k2xm+k2ym)/w2w0 - 0.5*1j*dt*cvphs[i]*(kxm*vgas[i][0] + kym*vgas[i][1]))*vas[i]
    return vas

def IntegrationStep(f0,ua,ub,vga,vgb,cvph1,cvph2,w2w0,kxm,kym,k2xm,k2ym,dt,Es,coupling):
    # taking FFT from initial envelopes
    va=np.fft.fft2(ua) 
    vb=np.fft.fft2(ub)
   
    #first half-step wrt vg d/dx term & diffraction term
    vna1=DiffractionHalfStep(va,vga,cvph1,1.0,kxm,kym,k2xm,k2ym,dt)
    una1=np.fft.ifft2(vna1);
   
    vnb1=DiffractionHalfStep(vb,vgb,cvph2,w2w0,kxm,kym,k2xm,k2ym,dt)
    unb1=np.fft.ifft2(vnb1);
    
    #applying self-focusing term
    pot=Es*((np.abs(una1))**2+(np.abs(unb1))**2);
    una2=np.exp(-1j*dt*pot)*una1;
    unb2=np.exp(-1j*dt*pot)*unb1;
    
    #handling source term due to density perturbations
    A11 = 1.0-(dt*np.abs(f0))**2/2.0; # sqrt(1.-(dt.*abs(f0)).^2) ;
    A12 = np.conjugate(f0)*dt;
    A21 = -1.0*f0*dt;
    A22 = 1.0-(dt*np.abs(f0))**2/2.0; # sqrt(1.-(dt.*abs(f0)).^2) ;   

    a = (A11*una2+A12*unb2);
    b = (A21*una2+A22*unb2);

    f = f0+dt*np.conjugate(a)*b*coupling

    una2 = a; 
    unb2 = b; 
    f0 = f; 
    
    #final half-step wrt vg d/dx term & diffraction term
    vna2=np.fft.fft2(una2);
    va=DiffractionHalfStep(vna2,vga,cvph1,1.0,kxm,kym,k2xm,k2ym,dt)
  
    vnb2=np.fft.fft2(unb2);
    vb=DiffractionHalfStep(vnb2,vgb,cvph2,w2w0,kxm,kym,k2xm,k2ym,dt)

    ua=np.fft.ifft2(va);
    ub=np.fft.ifft2(vb);
    return ua,ub,f0

# two-pump integrator

def IntegrationStepTwoPump(f0,g0,ua,ub,uc,vga,vgb,vgc,cvph1,cvph2,cvph3,w2w0,kxm,kym,k2xm,k2ym,dt,Es,coupling1,coupling2):
    # taking FFT from initial envelopes
    va=np.fft.fft2(ua)
    vb=np.fft.fft2(ub)
    vc=np.fft.fft2(uc)

    #first half-step wrt vg d/dx term & diffraction term
    vna1=DiffractionHalfStep(va,vga,cvph1,1.0,kxm,kym,k2xm,k2ym,dt)
    una1=np.fft.ifft2(vna1)

    vnb1=DiffractionHalfStep(vb,vgb,cvph2,w2w0,kxm,kym,k2xm,k2ym,dt)
    unb1=np.fft.ifft2(vnb1)

    vnc1=DiffractionHalfStep(vc,vgc,cvph3,w2w0,kxm,kym,k2xm,k2ym,dt)
    unc1=np.fft.ifft2(vnc1)

    #applying self-focusing term
    pot=Es*((np.abs(una1))**2+(np.abs(unb1))**2+(np.abs(unc1))**2)
    una2=np.exp(-1j*dt*pot)*una1
    unb2=np.exp(-1j*dt*pot)*unb1
    unc2=np.exp(-1j*dt*pot)*unc1

    #handling source term due to density perturbations
    A11 = 1.0-(dt*np.abs(f0))**2/2.0 - (dt*np.abs(g0))**2/2.0 # sqrt(1.-(dt.*abs(f0)).^2) ;
    A12 = np.conjugate(f0)*dt
    A13 = np.conjugate(g0)*dt
    A21 = -1.0*f0*dt
    A22 = 1.0-(dt*np.abs(f0))**2/2.0 # sqrt(1.-(dt.*abs(f0)).^2) ;
    A23 = -f0*np.conjugate(g0)*dt**2/2.0
    A31 = -g0*dt
    A32 = -np.conjugate(f0)*g0*dt**2/2.0
    A33 = 1.0 - (dt*np.abs(g0))**2/2.0

    a = (A11*una2+A12*unb2+A13*unc2)
    b = (A21*una2+A22*unb2+A23*unc2)
    c = (A31*una2+A32*unb2+A33*unc2)

    f = f0+dt*np.conjugate(a)*b*coupling1
    g = g0+dt*np.conjugate(a)*c*coupling2
 
    una2 = a
    unb2 = b
    unc2 = c
    f0 = f
    g0 = g

    #final half-step wrt vg d/dx term & diffraction term
    vna2=np.fft.fft2(una2);
    va=DiffractionHalfStep(vna2,vga,cvph1,1.0,kxm,kym,k2xm,k2ym,dt)

    vnb2=np.fft.fft2(unb2);
    vb=DiffractionHalfStep(vnb2,vgb,cvph2,w2w0,kxm,kym,k2xm,k2ym,dt)

    vnc2=np.fft.fft2(unc2);
    vc=DiffractionHalfStep(vnc2,vgc,cvph3,w2w0,kxm,kym,k2xm,k2ym,dt)

    ua=np.fft.ifft2(va)
    ub=np.fft.ifft2(vb)
    uc=np.fft.ifft2(vc)

    return ua,ub,uc,f0,g0



def IntegrationStepTwoPumpOneBeat(f0,ua,ub,uc,vga,vgb,vgc,cvph1,cvph2,cvph3,w2w0,kxm,kym,k2xm,k2ym,dt,Es,coupling1,coupling2):
    # taking FFT from initial envelopes
    va=np.fft.fft2(ua)
    vb=np.fft.fft2(ub)
    vc=np.fft.fft2(uc)

    #first half-step wrt vg d/dx term & diffraction term
    vna1=DiffractionHalfStep(va,vga,cvph1,1.0,kxm,kym,k2xm,k2ym,dt)
    una1=np.fft.ifft2(vna1)

    vnb1=DiffractionHalfStep(vb,vgb,cvph2,w2w0,kxm,kym,k2xm,k2ym,dt)
    unb1=np.fft.ifft2(vnb1)

    vnc1=DiffractionHalfStep(vc,vgc,cvph3,w2w0,kxm,kym,k2xm,k2ym,dt)
    unc1=np.fft.ifft2(vnc1)

    #applying self-focusing term
    pot=Es*((np.abs(una1))**2+(np.abs(unb1))**2+(np.abs(unc1))**2)
    una2=np.exp(-1j*dt*pot)*una1
    unb2=np.exp(-1j*dt*pot)*unb1
    unc2=np.exp(-1j*dt*pot)*unc1

    #handling source term due to density perturbations
    A11 = 1.0-(dt*np.abs(f0))**2/2.0-(dt*np.abs(f0))**2/2.0 # sqrt(1.-(dt.*abs(f0)).^2) ;
    A12 = np.conjugate(f0)*dt
    A13 = np.conjugate(f0)*dt
    A21 = -1.0*f0*dt
    A22 = 1.0-(dt*np.abs(f0))**2/2.0 # sqrt(1.-(dt.*abs(f0)).^2) ;
    A23 = -f0*np.conjugate(f0)*dt**2/2.0
    A31 = -f0*dt
    A32 = -np.conjugate(f0)*f0*dt**2/2.0
    A33 = 1.0 - (dt*np.abs(f0))**2/2.0

    a = (A11*una2+A12*unb2+A13*unc2)
    b = (A21*una2+A22*unb2+A23*unc2)
    c = (A31*una2+A32*unb2+A33*unc2)

    f = f0+dt*np.conjugate(a)*b*coupling1+dt*np.conjugate(a)*c*coupling2

    una2 = a
    unb2 = b
    unc2 = c
    f0 = f

    #final half-step wrt vg d/dx term & diffraction term
    vna2=np.fft.fft2(una2);
    va=DiffractionHalfStep(vna2,vga,cvph1,1.0,kxm,kym,k2xm,k2ym,dt)

    vnb2=np.fft.fft2(unb2);
    vb=DiffractionHalfStep(vnb2,vgb,cvph2,w2w0,kxm,kym,k2xm,k2ym,dt)

    vnc2=np.fft.fft2(unc2);
    vc=DiffractionHalfStep(vnc2,vgc,cvph3,w2w0,kxm,kym,k2xm,k2ym,dt)

    ua=np.fft.ifft2(va)
    ub=np.fft.ifft2(vb)
    uc=np.fft.ifft2(vc)

    return ua,ub,uc,f0


#multi-pump integrator

def IntegrationStepMultiPump(f0,uvec,vgvec,cvvec,w2w0,kxm,kym,k2xm,k2ym,dt,Es,couplings):
    # taking FFT from initial envelopes
    vvec=np.fft.fft2(uvec)

    #first half-step wrt vg d/dx term & diffraction term
    
    vna1=DiffractionHalfStep(vvec[0],vgvec[0],cvvec[0],1.0,kxm,kym,k2xm,k2ym,dt)
    una1=np.fft.ifft2(vna1)

    vnb1=DiffractionHalfStepPump(vvec[1:],vgvec[1:],cvvec[1:],w2w0,kxm,kym,k2xm,k2ym,dt)
    unb1=np.fft.ifft2(vnb1)

    #applying self-focusing term
    pot=(np.abs(una1))**2
    for unb in unb1:
        pot=pot+np.abs(unb)**2
    pot=Es*pot

    una2=np.exp(-1j*dt*pot)*una1
    for i in range(len(unb1)):
        unb1[i]=np.exp(-1j*dt*pot)*unb1[i]
    unb2=unb1

    unvec=[]
    unvec.append(una2)
    for unbenv in unb2:
        unvec.append(unbenv)

    #list representation of matrix for energy exchange between beams
    Alist=[]

    for i in range(len(uvec)**2):
        Alist.append([])

    Alist[0]=1-0.5*f0*np.conjugate(f0)*dt**2*len(unb2)

    for i in range(len(unb2)):
        Alist[i+1]=dt*np.conjugate(f0)
    
    for i in range(len(unb2)):
        Alist[len(uvec)*(i+1)]=-dt*f0

    for i in range(len(unb2)):
        Alist[len(uvec)*(i+1)+i+1]=np.ones(uvec[0].shape)      


    for i in range(len(unb2)):
        for j in range(len(unb2)):
            buff=Alist[len(unvec)*(i+1)+j+1]
            if len(buff)>=1:
                Alist[len(unvec)*(i+1)+j+1]=buff-0.5*f0*np.conjugate(f0)*dt**2
            else:
                Alist[len(unvec)*(i+1)+j+1]=-0.5*f0*np.conjugate(f0)*dt**2

    #multiply matrix exponent calculated in Alist by vector of envelopes
    Alist=np.array(Alist)
    unvec=np.array(unvec)

    avec=[]
    for i in range(len(unvec)):
        avec.append(sum(Alist[i*len(unvec):(i+1)*len(unvec),:,:]*unvec))

    #avec is the list of arrays of envelopes

    for i in range(len(unb2)):
        f0=f0+dt*np.conjugate(avec[0])*avec[i+1]*couplings[i]

    unvec2 = avec

    #final half-step wrt vg d/dx term & diffraction term
    vvec=np.fft.fft2(unvec2)

    vna1=DiffractionHalfStep(vvec[0],vgvec[0],cvvec[0],1.0,kxm,kym,k2xm,k2ym,dt)
    una1=np.fft.ifft2(vna1)

    vnb1=DiffractionHalfStepPump(vvec[1:],vgvec[1:],cvvec[1:],w2w0,kxm,kym,k2xm,k2ym,dt)
    unb1=np.fft.ifft2(vnb1)

    #collect list of envelopes to dump
    uvec=[]
    uvec.append(una1)
    for unbenv in unb1:
        uvec.append(unbenv)

    return uvec,f0

#multi-pump multi-beat integrator

def IntegrationStepMultiPumpMultiBeat(f0s,uvec,vgvec,cvvec,w2w0,kxm,kym,k2xm,k2ym,dt,Es,couplings):
    # taking FFT from initial envelopes
    vvec=np.fft.fft2(uvec)

    #first half-step wrt vg d/dx term & diffraction term

    vna1=DiffractionHalfStep(vvec[0],vgvec[0],cvvec[0],1.0,kxm,kym,k2xm,k2ym,dt)
    una1=np.fft.ifft2(vna1)

    vnb1=DiffractionHalfStepPump(vvec[1:],vgvec[1:],cvvec[1:],w2w0,kxm,kym,k2xm,k2ym,dt)
    unb1=np.fft.ifft2(vnb1)

    #applying self-focusing term
    pot=(np.abs(una1))**2
    for unb in unb1:
        pot=pot+np.abs(unb)**2
    pot=Es*pot

    una2=np.exp(-1j*dt*pot)*una1
    for i in range(len(unb1)):
        unb1[i]=np.exp(-1j*dt*pot)*unb1[i]
    unb2=unb1

    unvec=[]
    unvec.append(una2)
    for unbenv in unb2:
        unvec.append(unbenv)

    #list representation of matrix for energy exchange between beams
    Alist=[]

    for i in range(len(uvec)**2):
        Alist.append([])

    Alist[0]=1.0

    for f0 in f0s:
        Alist[0]=Alist[0]-0.5*f0*np.conjugate(f0)*dt**2

    for i in range(len(unb2)):
        Alist[i+1]=dt*np.conjugate(f0s[i])

    for i in range(len(unb2)):
        Alist[len(uvec)*(i+1)]=-dt*f0s[i]

    for i in range(len(unb2)):
        Alist[len(uvec)*(i+1)+i+1]=np.ones(uvec[0].shape)


    for i in range(len(unb2)):
        for j in range(len(unb2)):
            buff=Alist[len(unvec)*(i+1)+j+1]
            if len(buff)>=1:
                Alist[len(unvec)*(i+1)+j+1]=buff-0.5*f0s[i]*np.conjugate(f0s[j])*dt**2
            else:
                Alist[len(unvec)*(i+1)+j+1]=-0.5*f0s[i]*np.conjugate(f0s[j])*dt**2

    #multiply matrix exponent calculated in Alist by vector of envelopes
    Alist=np.array(Alist)
    unvec=np.array(unvec)

    avec=[]
    for i in range(len(unvec)):
        avec.append(sum(Alist[i*len(unvec):(i+1)*len(unvec),:,:]*unvec))

    #avec is the list of arrays of envelopes

    for i in range(len(unb2)):
        f0s[i]=f0s[i]+dt*np.conjugate(avec[0])*avec[i+1]*couplings[i]

    unvec2 = avec

    #final half-step wrt vg d/dx term & diffraction term
    vvec=np.fft.fft2(unvec2)
    
    vna1=DiffractionHalfStep(vvec[0],vgvec[0],cvvec[0],1.0,kxm,kym,k2xm,k2ym,dt)
    una1=np.fft.ifft2(vna1)
    
    vnb1=DiffractionHalfStepPump(vvec[1:],vgvec[1:],cvvec[1:],w2w0,kxm,kym,k2xm,k2ym,dt)
    unb1=np.fft.ifft2(vnb1)
    
    #collect list of envelopes to dump
    uvec=[]
    uvec.append(una1)
    for unbenv in unb1:
        uvec.append(unbenv)

    return uvec,f0s

