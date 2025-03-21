import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

from cosmology import background, Weyl_power_spectra


def get_cls_mixed_LOS_os(b, chimax, lmax, nl):
    """
    This function generates Cls for convergence and shear
    It takes as argument the maximum multipole lmax.
    nl is the number of values to be computed.

    b : redshift bin in question (0 to 4)
    """

    get_item('W_LOS_mean_intp', 'WW_LOS_rms_intp', 'W_os_mean_intp', 'WW_os_rms_intp')
    
    nz = 100 #number of elements for discrete integral along the los
    
    # Conformal distances and redshifts
    # results = camb.get_background(pars)
    chis = np.linspace(0, chimax, nz)
    zs = background.redshift_at_comoving_radial_distance(chis)
    
    # Array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]
    
    # Lensing kernel (here LOS shear)
    kernel2LOS = W_LOS_mean_intp(chis) / chis
    kernelLOS = WW_LOS_rms_intp(chis) / chis
    
    # Lensing kernel (here weak lensing shear)
    kernel2os = W_os_mean_intp[b](chis) / chis 
    kernelos = WW_os_rms_intp[b](chis) / chis
    
    # Integration over chi
    lmin = 1
    ls = np.logspace(np.log10(lmin), np.log10(lmax), nl)
    cl2LOSos = np.zeros(ls.shape)   
    cl32LOSos2 = np.zeros(ls.shape)
    cl32LOS2os = np.zeros(ls.shape)
    cl1LOSos = np.zeros(ls.shape)
    w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
    
    # Check that lmax is not too big
    if lmax > extrap_kmax * chis[-1]:
        print("""Warning: lmax is too large given the range of extrapolation given to CAMB
for the power spectrum. The results cannot be trusted.""")
    
    for i, l in enumerate(ls):
        k = (l + 0.5)/chis
        w[:] = 1
        w[k<1e-4] = 0
        w[k>=extrap_kmax] = 0
        
        #possibly wrong, need to check these kernels carefully 
        cl2LOSos[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel2LOS * kernel2os)   
        cl32LOSos2[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernelLOS * kernel2os) 
        cl32LOS2os[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel2LOS * kernelos)       
        cl1LOSos[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernelLOS * kernelos)
        
    return ls, cl2LOSos, cl32LOSos2, cl32LOS2os, cl1LOSos

