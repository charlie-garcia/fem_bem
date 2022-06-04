#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: ibah, carlos
"""
#==========================================================================================================
#                                                                                                           #
#                                           20/09/2021                                                      #
#                                   Absorption    coefficent of circular membrane                           #
#                                                                                                           #
#============================================================================================================

import numpy as np 
import sys
from matplotlib import pyplot as plt 
import scipy.special as sp

#%% Create Geometry
# geometrical caracteristic of membrane 
a, xc , yc, zc, NN1  =  0.3/2, 0, 0, 0, 0.008       # radius, center x, center y, center z, resolution

#%% Analtical solution
T, h, rho = 800, 254e-6, 1380                       # T = tension, H = thickness, rho = volumic density
rho_s     = rho*h                                   # rho_s = surfacic density

m = 0
n = 0
xi      = sp.jn_zeros(m, n+1)
w_n     = xi/a * np.sqrt(T/rho_s)
mass_mn = rho_s * np.pi*a**2*( sp.jv(m+1,xi) )**2
PHI     = ( 2* np.pi*a**2*sp.jv(m+1,xi)/xi )**2
S       = np.pi*a**2 

#%% Acoustics
c_0, rho_0 = 340,   1.25                            # c air, \rho0 air
Z0         = rho_0 *c_0

fmin, fmax, Nf = 1, 700, 700

eta_n = [0.1, .08, 0.05]                           # damping values
freq  = np.linspace(fmin, fmax, Nf)
alpha =  np.zeros((Nf, len(eta_n)))

# Compute absorption coeff
for kk in range(Nf):
    omega = 2*np.pi*freq[kk]
    ii=0
    for eta in eta_n:
        H01 = 1j*omega/( mass_mn*(w_n**2*(1+1j*eta) - omega**2))
        R   = (1 - (Z0*H01/S)*PHI) /(1 + (Z0*H01/S)*PHI)
        alpha[kk, ii] = 1-(abs(R))**2
        ii+=1

#%% Plotting alpha for the three values of eta    
fig, (ax1,ax2) = plt.subplots(2,1)

for ii in range(len(eta_n)):
    ax1.plot(freq, alpha[:,ii], "-",  lw=2, ms=12, label="$\eta =%.2f$"%(eta_n[ii]))
    ax2.plot(freq, alpha[:,ii], "-",  lw=2, ms=12, label="$\eta =%.2f$"%(eta_n[ii]))

plt.sca(ax1)
plt.vlines(w_n/2/np.pi, 0,1, 'gray', alpha = 0.5)
plt.axis([freq[0], freq[-1], 0, np.max(alpha)*1.1])
plt.xlabel("Fréquence (Hz)")
plt.ylabel(r"$\alpha$")
plt.legend()
plt.grid(axis='both',linestyle=':', color='k')


plt.sca(ax2)
plt.vlines(w_n/2/np.pi, 0,1, 'gray', alpha = 0.5)
plt.axis([freq[0], freq[-1], 0,1])
plt.xlabel("Fréquence (Hz)")
plt.ylabel(r"$\alpha$")
plt.legend()
plt.grid(axis='both',linestyle=':', color='k')

plt.show()



