import numpy as np
from config import extrap_kmax
from functions import useful_functions as uf 
from cosmology import background, Weyl_power_spectra, correlations_prefactor


def get_cls_mixed_d_os(b1, b2, chimax, lmax, nl):
    """
    This function generates Cls for convergence and shear
    It takes as argument the maximum multipole lmax.
    nl is the number of values to be computed.

    b1 : d redshift bin in question (0 to 4)
    b2 : os redshift bin in question (0 to 4)
    """
    
    uf.get_item('W_d_mean_intp', 'WW_d_rms_intp', 'W_os_mean_intp', 'WW_os_rms_intp')
    
    nz = 100 #number of elements for discrete integral along the los
    
    # Conformal distances and redshifts
    # results = camb.get_background(pars)
    chis = np.linspace(0, chimax, nz)
    zs = background.redshift_at_comoving_radial_distance(chis)
    
    # Array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]
    
    # Lensing kernel (here position)
    kernel2d = W_d_mean_intp[b1](chis) / chis
    kerneld = WW_d_rms_intp[b1](chis) / chis  
    
    # Lensing kernel (here weak lensing shear)
    kernel2os = W_os_mean_intp[b2](chis) / chis
    kernelos = WW_os_rms_intp[b2](chis) / chis
    
    # Integration over chi
    lmin = 1
    ls = np.logspace(np.log10(lmin), np.log10(lmax), nl)
    cl2dos = np.zeros(ls.shape)
    cl32dos2 = np.zeros(ls.shape)  
    cl32d2os = np.zeros(ls.shape)
    cl1dos = np.zeros(ls.shape)
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
        
        cl2dos[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel2os * kernel2d)    * correlations_prefactor
        cl32dos2[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kerneld * kernel2os) * correlations_prefactor
        cl32d2os[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel2d * kernelos) * correlations_prefactor
        cl1dos[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernelos * kerneld) * correlations_prefactor
        
    return ls, cl2dos, cl32dos2, cl32d2os, cl1dos