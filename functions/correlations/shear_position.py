import numpy as np
from config import extrap_kmax
from functions import useful_functions as uf 
from cosmology import background, Weyl_power_spectra, correlations_prefactor


def get_cls_mixed_LOS_d(b, chimax, lmax, nl):
    """
    This function generates Cls for convergence and shear
    It takes as argument the maximum multipole lmax.
    nl is the number of values to be computed.

    b : redshift bin in question (0 to 4)
    """

    uf.get_item('W_LOS_mean_intp', 'WW_LOS_rms_intp', 'W_d_intp', 'WW_d_rms_intp')
    
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
    
    # Lensing kernel (here position)
    kernel2d = W_d_intp[b](chis) / chis
    kerneld = WW_d_rms_intp[b](chis) / chis
    
    kernel2d = np.heaviside(kernel2d,0)*kernel2d
    kernel2d = np.sqrt(kernel2d)
		
    # Integration over chi
    lmin = 1
    ls = np.logspace(np.log10(lmin), np.log10(lmax), nl)
    cl2LOSd = np.zeros(ls.shape)   
    cl32LOSd2 = np.zeros(ls.shape)
    cl32LOS2d = np.zeros(ls.shape)
    cl1LOSd = np.zeros(ls.shape)
    w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
		
    # Check that lmax is not too big
    if lmax > extrap_kmax * chis[-1]:
        print("""Warning: lmax is too large given the range of extrapolation given to CAMB for the power spectrum. The results cannot be trusted.""")
		
    for i, l in enumerate(ls):
        k = (l + 0.5)/chis
        w[:] = 1
        w[k<1e-4] = 0
        w[k>=extrap_kmax] = 0
        power = w * Weyl_power_spectra.P(zs, k, grid=False)
        cl2LOSd[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel2LOS * kernel2d) * correlations_prefactor  
        cl32LOSd2[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernelLOS * kernel2d) * correlations_prefactor
        cl32LOS2d[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel2LOS * kerneld) * correlations_prefactor
        cl1LOSd[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernelLOS * kerneld) * correlations_prefactor

    return ls, cl2LOSd, cl32LOSd2, cl32LOS2d, cl1LOSd