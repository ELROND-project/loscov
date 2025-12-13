import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('redshift_distributions')

def get_cls_mixed_EP(b1, b2, chimax, lmax, nl):
    """
    This function generates Cls for convergence and shear
    It takes as argument the maximum multipole lmax.
    nl is the number of values to be computed.

    b1 : os redshift bin in question (0 to Nbinz_E)
    b2 : d redshift bin in question (0 to Nbinz_P)
    """
    
    get_item('Q_d_mean_intp', 'QQ_d_rms_intp', 'Q_os_mean_intp', 'QQ_os_rms_intp')
    
    nz = 100 #number of elements for discrete integral along the los
    
    # Conformal distances and redshifts
    results = camb.get_background(pars)
    chis = np.linspace(0, chimax, nz)
    zs = results.redshift_at_comoving_radial_distance(chis)
    
    # Array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]

    #CAMB correction
    CAMB_factor = ((1+zs) * 1.5 * Omega_M * (H0/(c*1e-3))**2)**(-1)
    
    # Lensing kernel (here weak lensing shear)
    kernel2os = Q_os_mean_intp[b1](chis) * CAMB_factor
    kernelos = QQ_os_rms_intp[b1](chis) * CAMB_factor
    
    # Lensing kernel (here position)
    kernel2d = Q_d_mean_intp[b2](chis) * CAMB_factor
    kerneld = QQ_d_rms_intp[b2](chis)  * CAMB_factor
    
    
    # Integration over chi
    lmin = 1
    ls = np.logspace(np.log10(lmin), np.log10(lmax), nl)
    cl2dos = np.zeros(ls.shape)
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
        
        cl2dos[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel2os * kernel2d) 
        
    return ls, cl2dos