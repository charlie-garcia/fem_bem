import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as inter
# from fenics import *

def integralPath(x,y,u):
    z = np.sqrt(x**2 + y**2)
  
    f = inter.interp1d(z, u, kind='cubic')
    new_z = np.linspace(z.min(), z.max())
    new_u  = f(new_z)
     
    nds = new_z[1]-new_z[0]
    sol = np.trapz(new_u, dx = nds)
 
    return sol

def check_array(x):
    try:
        x.shape
        return True
    except:
        return False
        
def dbW(signal):
	sdB = 10*np.log10(np.abs(signal)/10e-12)
	return sdB
