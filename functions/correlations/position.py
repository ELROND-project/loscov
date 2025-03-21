import numpy as np
from config import extrap_kmax
from functions.galaxy_distribution import pb
from functions import useful_functions as uf 
from cosmology import background, Weyl_power_spectra, correlations_prefactor

#################################################### Galaxy bias ######################################################

#the galaxy bias
def bias(z):
    return 1.1*z**2.4/(1+z)+0.9

#the "weight function" (the equivalent of W_mean in os and LOS)
def W_d(chi, b):

    redshift = background.redshift_at_comoving_radial_distance(chi)
    
    return pb(redshift, b) * bias(redshift) / (1+redshift)

def WW_d(chi, b):
    
    return W_d(chi, b)**2   
    

################################################ Getting cls #####################################################

def get_cl_d(b1, b2, chimax, lmax, nl):
    """
    This function generates Cls for the integrated matter correlation function
    It takes as argument the maximum multipole lmax.
    nl is the number of values to be computed.
    b1 and b2 are the considered redshift bins.
    returns an array of ls and the corresponding array of Cls
    """

    uf.get_item('W_d_intp', 'WW_d_rms_intp') #these are just interpolated versions of the weight functions
    
	# Integration over chi
    lmin = 1
    ls = np.logspace(np.log10(lmin), np.log10(lmax), nl)
    integral = np.zeros(ls.shape)
    integral2 = np.zeros(ls.shape)
    integral32 = np.zeros(ls.shape)
    
    if b1 != b2:         #this is relevant for positions, but not for shapes
    	return ls, integral2, integral, integral32
        
    else:
        nz = 100 #number of elements for discrete integral along the los

        # Conformal distances and redshifts
        # results = camb.get_background(pars)
        chis = np.linspace(0, chimax, nz)
        zs = background.redshift_at_comoving_radial_distance(chis)

		# Array of delta_z, and drop first and last points where things go singular
        dchis = (chis[2:]-chis[:-2])/2
        chis = chis[1:-1]
        zs = zs[1:-1]

		# Everything in the integrand except Weyl power spectrum
        kernel2 = W_d_intp[b1](chis) * W_d_intp[b2](chis) / chis**2
        kernel1 = WW_d_rms_intp[b1](chis) * WW_d_rms_intp[b2](chis) / chis**2 
        kernel32 = W_d_intp[b1](chis) * WW_d_rms_intp[b2](chis) / chis**2         #this is almost certainly wrong, just left in here as a stop gap - not currently being used, but will be relevant for later work

        w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation

		# Check that lmax is not too big
        if lmax > extrap_kmax * chis[-1]: # given the value of z_max, l_max can go up to 1e13 and it's fine
            print("""Warning: lmax is too large given the range of extrapolation given to CAMB
		for the power spectrum. The results cannot be trusted.""")

        for i, l in enumerate(ls):
            k = (l + 0.5)/chis
            w[:] = 1
            w[k<1e-4] = 0
            w[k>=extrap_kmax] = 0
            integral2[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel2)
            integral[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel1)
            integral32[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel32)

        cl1 = integral * correlations_prefactor**2
        cl2 = integral2 * correlations_prefactor**2
        cl32 = integral32 * correlations_prefactor**2

        return ls, cl2, cl1, cl32