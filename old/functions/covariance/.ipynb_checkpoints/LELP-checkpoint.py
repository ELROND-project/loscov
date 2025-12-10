import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from galaxy_distribution import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp','LLt','LP')

############################################################## 6.2 LLLP ######################################################################
##############################################################################################################################################

################################################## 6.2.1 LLLP cosmic covariance ##############################################################

def generate_ccov_LLLP(distributions, D, approx=False):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - galaxy position correlation functions.

    lens_dist      : statistics relating to the lens distribution
    galaxy_dist    : statistics relating to the galaxy distribution
    D             : the galaxy redshift bin D (0 to 4)
    """
    
    distribution1 = distributions['LL']
    distribution2 = distributions['Lp']
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for LL 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for LL (in rad)
    
    Nbin2       = distribution2.Nbina      #the number of angular bins for Lp 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Lp (in rad)
    
    # Initialise the blocks
    ccov = np.zeros((Nbin, Nbin))
    
    # Define the integrands (complete from here)
    
    def integrand_p(params):
        
        psi_j, psi_kd, r_b, r_kd, r_k = params
    
        y_kj = r_j*np.sin(psi_j)
        x_kj = r_j*np.cos(psi_j) - r_k
        
        r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
        psi_kj = np.arctan2(y_kj, x_kj)
        
        r_jd = cos_law_side(r_kd, r_kj, (psi_kd-psi_kj))
        psi_jd = cos_law_angle(r_kd, r_jd, r_kj) + psi_kd

        f = LP[D](r_jd) * cos2(psi_jd - psi_j)
            * ( LLp(r_k) * cos2(psi_j) * cos2(psi_kd)
              + LLt(r_k) * sin2(psi_j) * sin2(psi_kd)
              )

        f *= 2 * np.pi * r_k
        
        return f
    
    def integrand_t(params):
        
        psi_j, psi_kd, r_b, r_kd, r_k = params
    
        y_kj = r_j*np.sin(psi_j)
        x_kj = r_j*np.cos(psi_j) - r_k
        
        r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
        psi_kj = np.arctan2(y_kj, x_kj)
        
        r_jd = cos_law_side(r_kd, r_kj, (psi_kd-psi_kj))
        psi_jd = cos_law_angle(r_kd, r_jd, r_kj) + psi_kd

        f = LP[D](r_jd) * cos2(psi_jd - psi_j)
            * ( LLp(r_k) * sin2(psi_j) * cos2(psi_kd)
              - LLt(r_k) * cos2(psi_j) * sin2(psi_kd)
              )

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
    
    for alpha in range(Nbin1):
        for beta in range(Nbin2): 
                     
            ccov_p[alpha, beta], err_p = integral_bins(integrand_p, alpha, beta, Csamp)
            ccov_t[alpha, beta], err_t = integral_bins(integrand_t, alpha, beta, Csamp)

            test_err(err_p, ccov_p[alpha, beta], f'LLLp ccov plus redshifts {D} angular bins {alpha, beta}')
            test_err(err_t, ccov_t[alpha, beta], f'LLLp ccov times redshifts {D} angular bins {alpha, beta}')
            
    # Make the full cosmic covariance matrix (need to fix this to make plus and minus)

    ccov = np.block([[ccov_p],
                     [ccov_t]])
    
    return ccov

################################################## 6.2.2 LeLe noise/sparsity covariance #############################################################

def generate_ncov_LLLP(delta_BD, sigma_LOS, distributions, B, D, approx=False):
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
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for LL 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for LL (in rad)
    
    Nbin2       = distribution2.Nbina      #the number of angular bins for Lp 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Lp (in rad)
    
    # Initialise the blocks
    ncov = np.zeros((Nbin, Nbin))
    
    # Define the integrands
    
    def integrand_p(params):
        
        r_i, r_d, psi_d = params
    
        y_id = r_d*np.sin(psi_d)
        x_id = r_d*np.cos(psi_d) - r_i
        
        r_id = np.sqrt( y_id**2 + x_id**2 ) 
        psi_id = np.arctan2(y_id, x_id)
        
        f = LP[D](r_id) * cos2(psi_d) * cos2(psi_id) 

        f *= 2 * np.pi * r_i
                              
        return f
    
    def integrand_t(params):
        
        r_i, r_d, psi_d = params
    
        y_id = r_d*np.sin(psi_d)
        x_id = r_d*np.cos(psi_d) - r_i
        
        r_id = np.sqrt( y_id**2 + x_id**2 ) 
        psi_id = np.arctan2(y_id, x_id)
        
        f = LP[D](r_id) * sin2(psi_d) * sin2(psi_id) 

        f *= 2 * np.pi * r_i
                              
        return f
    
    def integral_bins(integrand, alpha, beta, nsampl):
        
        ranges = [rs[alpha], rs[alpha+1]), (rs[beta], rs[beta+1]), (0, 2*np.pi)]
        
        integral, err = monte_carlo_integrate(integrand, ranges, nsampl)
        
        # normalisation of differential elements
        integral *= (sigma_LOS**2/Nlens) / (Omegas[alpha] * Omegas[beta]) 
        err      *= (sigma_LOS**2/Nlens) / (Omegas[alpha] * Omegas[beta]) 
        
        return integral, err
    
    for alpha in range(Nbin1):
        for beta in range(Nbin2): 
                     
            ncov_p[alpha, beta], err_p = integral_bins(integrand_p, alpha, beta, Csamp)
            ncov_t[alpha, beta], err_t = integral_bins(integrand_t, alpha, beta, Csamp)

            test_err(err_p, ncov_p[alpha, beta], f'LLLp ncov plus redshifts {D} angular bins {alpha, beta}')
            test_err(err_t, ncov_t[alpha, beta], f'LLLp ncov times redshifts {D} angular bins {alpha, beta}')
            
    # Make the full cosmic covariance matrix (need to fix this to make plus and minus)

    ncov = np.block([[ncov_p],
                     [ncov_t]])
    
    return ncov