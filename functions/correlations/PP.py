import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('redshift_distributions')

#################################################### Galaxy bias ######################################################

#the galaxy bias
def bias(z):
    return 1.1*z**2.4/(1+z)+0.9

#the "weight function" (the equivalent of W_mean in os and LOS)
def Q_d(chi, b):

    redshift = background.redshift_at_comoving_radial_distance(chi)

    prefactor =  (H0/(c*1e-3))*np.sqrt(Omega_M*(1+redshift)**3+Omega_L)
    
    return prefactor * redshift_distributions['P'].pb(redshift, b) * bias(redshift) / chi

def QQ_d(chi, b):
    
    return Q_d(chi, b)**2   
    

################################################ Getting cls #####################################################

def get_cl_P(b1, b2, chimax, lmax, nl):
    """
    This function generates cls for the integrated matter correlation function
    It takes as argument the maximum multipole lmax.
    nl is the number of values to be computed.
    b1 and b2 are the considered redshift bins (0 to Nbinz_P).
    returns an array of ls and the corresponding array of cls
    """

    get_item('Q_d_intp', 'QQ_d_rms_intp') #these are just interpolated versions of the weight functions
    
	# Integration over chi
    lmin = 1
    ls = np.logspace(np.log10(lmin), np.log10(lmax), nl)
    integral2 = np.zeros(ls.shape)
    
    if b1 != b2:         #this is relevant for positions, but not for shapes
    	return ls, integral2
        
    else:
        nz = 100 #number of elements for discrete integral along the los

        # Conformal distances and redshifts
        results = camb.get_background(pars)
        chis = np.linspace(0, chimax, nz)
        zs = results.redshift_at_comoving_radial_distance(chis)

		# Array of delta_z, and drop first and last points where things go singular
        dchis = (chis[2:]-chis[:-2])/2
        chis = chis[1:-1]
        zs = zs[1:-1]

        #CAMB correction
        CAMB_factor = ((1+zs) * 1.5 * Omega_M * (H0/(c*1e-3))**2)**(-1)

		# Everything in the integrand except Weyl power spectrum
        kernel2 = Q_d_intp[b1](chis) * Q_d_intp[b2](chis) * CAMB_factor**2 

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
            
        cl2 = integral2

        return ls, cl2