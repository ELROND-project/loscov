import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from galaxy_distribution import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp','LLt','LP')

############################################################## 6.2 LPLP ######################################################################
##############################################################################################################################################

################################################## 6.2.1 LPLP cosmic covariance ##############################################################

def generate_ccov_LPLP(distributions, B, D, approx=False):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - galaxy position correlation functions.

    lens_dist      : statistics relating to the lens distribution
    galaxy_dist    : statistics relating to the galaxy distribution
    B             : the galaxy redshift bin B  (0 to 4)
    D             : the galaxy redshift bin D (0 to 4)
    """
    
    distribution = distributions['LP']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)
    
    # Initialise the blocks
    ccov = np.zeros((Nbin, Nbin))
    
    # Define the integrands (complete from here)
    
    def integrand(params):
        
        psi_b, psi_kd, r_b, r_kd, r_k = params
    
        y_kb = r_b*np.sin(psi_b)
        x_kb = r_b*np.cos(psi_b) - r_k
        
        r_kb = np.sqrt( y_kb**2 + x_kb**2 ) 
        psi_kb = np.arctan2(y_kb, x_kb)
        
        r_bd = cos_law_side(r_kd, r_kb, (psi_kd-psi_kb))
        psi_bd = cos_law_angle(r_kd, r_bd, r_kb) + psi_kd
        
        f = (LLp(r_k) * cos2(psi_b) * cos2(psi_kd)
            + LLt(r_k) * sin2(psi_b) * sin2(psi_kd))
            * PP[B][D](r_bd)
            + LP[D](r_bd) * PL[B](r_k) * cos2(psi_bd-psi_b) * cos2(psi_kd)

        f *= 2 * np.pi * r_k
        
        return f
    
    def integral_bins(integrand, alpha, beta, nsampl):
        
        ranges = [(0, 2*np.pi), (0, 2*np.pi),
                  (rs[alpha], rs[alpha+1]), (rs[beta], rs[beta+1]), (0, r2_max)]
        
        integral, err = monte_carlo_integrate(integrand, ranges, nsampl)
        
        # normalisation of differential elements
        integral /= (Omegatot * Omegas[alpha] * Omegas[beta]) 
        err /= (Omegatot * Omegas[alpha] * Omegas[beta]) 
        return integral, err
    
    for alpha in range(Nbin):
        for beta in range(alpha, Nbin): 
                     
            ccov[alpha, beta], err = integral_bins(integrand, alpha, beta, Csamp)
            ccov[beta, alpha] = ccov[alpha, beta]

            test_err(err, ccov[alpha, beta], f'LpLp ccov redshifts {B, D} angular bins {alpha, beta}')
            
    
    # Make the full cosmic covariance matrix

    ccov = np.block([[ccov]])
    
    return ccov

################################################## 6.2.2 LeLe noise/sparsity covariance #############################################################

def generate_ncov_LPLP(delta_BD, sigma_LOS, distributions, B, D, approx=False):
    """
    Computes the contribution of noise and sparsity variance in the 
    covariance matrix of the LOS shear - galaxy position correlation functions.

    delta_BD      : 
    sigma_LOS     : 
    lens_dist     : statistics relating to the lens distribution
    galaxy_dist   : statistics relating to the galaxy distribution
    B             : the galaxy redshift bin B  (0 to 4)
    D             : the galaxy redshift bin D (0 to 4)
    """
    
    distribution = distributions['LP']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)
    
    G_B    = get_ngal(B)         #G_B in the math - the number of galaxies in redshift bin B
    
    # Initialise the blocks
    ncov = np.zeros((Nbin, Nbin))
    
    # Define the integrands
    
    def integrand(params):
        
        r_i, r_k, psi_k = params
    
        y_ik = r_k*np.sin(psi_k)
        x_ik = r_k*np.cos(psi_k) - r_i
        
        r_ik = np.sqrt( y_ik**2 + x_ik**2 ) 
        psi_ik = np.arctan2(y_ik, x_ik)
        
        f = (delta_BD/G_B) * ( LLp(r_ik) * cos2(psi_ik) * cos2(psi_ik - psi_k) 
                             + LLt(r_ik) * sin2(psi_ik) * sin2(psi_ik - psi_k) )
            + (1/2) * (sigma_LOS**2 / Nlens) * PP[B][D](r_ik) * cos2(psi_k)

        f *= 2 * np.pi * r_i
                              
        return f
    
    def integral_bins(integrand, alpha, beta, nsampl):
        
        ranges = [rs[alpha], rs[alpha+1]), (rs[beta], rs[beta+1]), (0, 2*np.pi)]
        
        integral, err = monte_carlo_integrate(integrand, ranges, nsampl)
        
        # normalisation of differential elements
        integral /= (Omegas[alpha] * Omegas[beta]) 
        err     /= (Omegas[alpha] * Omegas[beta]) 
        
        return integral, err
    
    for alpha in range(Nbin):
        for beta in range(alpha, Nbin): 
                     
            ncov[alpha, beta], err = integral_bins(integrand, alpha, beta, Nsamp)
            ncov[beta, alpha] = ncov[alpha, beta]

            test_err(err, ncov[alpha, beta], f'LpLp ncov redshifts {B, D} angular bins {alpha, beta}')
            
    
    # Make the full cosmic covariance matrix

    ncov = np.block([[ncov]])
    
    return ccov