import numpy as np
from scipy import integrate

from config import binparams, extrap_kmax
from functions.galaxy_distribution import pb
from functions import useful_functions as uf 
from cosmology import background, Weyl_power_spectra

#################################################### d Weight Function ######################################################

def W_os(chi, chis):
    """
    weak lensing weight function, which tells us the relative contribution of matter
    at chi to the weak lensing of a source at chis

    chi : an inputted value of comoving distance
    chis : the comoving distance to the source
    """
    os = chi * (chis - chi) / chis #the weight function for gamma_os

    #the actual weight function
    W  = os * np.heaviside(os, 0)    #returns 0 if os is negative
    
    return W

def W_os_mean(chi, b):
    """
    Redshift-averaged os weight function

    chi : an inputted comoving distance
    b   : the redshift bin in question
    """
    
    z_min = binparams['redshifts'][b]                                  #the minimum redshift of a source
    z_max = binparams['redshifts'][b+1]                                #the maximum redshift of a source

    #this integrand returns the probability of a source being at redshift z_source
    def integrand(z_source):
    
        p_b = pb(z_source, b)                                        #the probability associated with a source being at z_source (in redshift bin b)
        chi_source = background.comoving_radial_distance(z_source)   #the comoving distance to the source at z_source
        
        return p_b*W_os(chi,chi_source) 

    #we integrate our weighting function over all the source positions in the relevant bin
    W, err = integrate.quad(integrand,z_min, z_max)
    
    return W

def WW_os_mean(chi, b):
    
    """
    Redshift-averaged os weight function

    chi : an inputted comoving distance
    b   : the redshift bin in question
    """
    
    z_min = binparams['redshifts'][b]                                  #the minimum redshift of a source
    z_max = binparams['redshifts'][b+1]                                #the maximum redshift of a source

    #this integrand returns the probability of a source being at redshift z_source
    def integrand(z_source):
    
        p_b = pb(z_source, b)                                        #the probability associated with a source being at z_source (in redshift bin b)
        chi_source = background.comoving_radial_distance(z_source)   #the comoving distance to the source at z_source
        
        return p_b*W_os(chi,chi_source)*W_os(chi,chi_source) 
    
    #we integrate our weighting function over all the source positions in the relevant bin
    W, err = integrate.quad(integrand,z_min, z_max)
    
    return W

def get_cl_gamma(b1, b2, chimax, lmax, nl):
    """
    This function generates Cls for convergence and shear
    It takes as argument the maximum multipole lmax.
    nl is the number of values to be computed.

    b  : redshift bin in question (1 to 5)
    """

    uf.get_item('W_os_mean_intp', 'WW_os_rms_intp')
    
    nz = 100 #number of elements for discrete integral along the los
    
    # Conformal distances and redshifts
    # results = camb.get_background(pars)
    chis = np.linspace(0, chimax, nz)
    zs = background.redshift_at_comoving_radial_distance(chis)
    
    # Array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]
    
    # Lensing kernel (here weak lensing shear)    
    kernel2 = W_os_mean_intp[b1](chis)*W_os_mean_intp[b2](chis) / chis**2
    kernel1 = WW_os_rms_intp[b1](chis)*WW_os_rms_intp[b2](chis) / chis**2 
    
    # Integration over chi
    lmin = 1
    ls = np.logspace(np.log10(lmin), np.log10(lmax), nl)
    integral = np.zeros(ls.shape)
    integral2 = np.zeros(ls.shape)
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
        power = w * Weyl_power_spectra.P(zs, k, grid=False) 
        integral[i] = np.dot(dchis, power * kernel1)
        integral2[i] = np.dot(dchis, power * kernel2)
        # The k**4 comes from the convention of CAMB for the Weyl potential, which is k**2 times
        # the actual gravitational potential.

    cl_os = integral
    cl_os2 = integral2
    
    return ls, cl_os2, cl_os