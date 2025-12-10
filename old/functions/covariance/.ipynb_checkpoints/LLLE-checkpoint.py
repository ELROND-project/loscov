import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from galaxy_distribution import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp','LLt')

############################################################## 6.2 LLLP ######################################################################
##############################################################################################################################################

################################################## 6.2.1 LLLP cosmic covariance ##############################################################

def generate_ccov_LLLL(distributions, approx=False):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear correlation functions.

    lens_dist      : statistics relating to the lens distribution
    """
    
    distribution = distributions['LL']
    
    Omegatot    = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin       = distribution.Nbina      #the number of angular bins for LL 
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits for LL (in rad)
    
    # Initialise the blocks
    ccov = np.zeros((Nbin, Nbin))
    
    # Define the integrands (complete from here)
    
    def integrand_pp(params):
        
        psi_j, psi_kl, r_j, r_kl, r_k = params
    
        y_kj = r_j*np.sin(psi_j)
        x_kj = r_j*np.cos(psi_j) - r_k
        
        r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
        psi_kj = np.arctan2(y_kj, x_kj)
        
        r_jl = cos_law_side(r_kl, r_kj, (psi_kl-psi_kj))
        psi_jl = cos_law_angle(r_kl, r_jl, r_kj) + psi_kl

        f = ( LLp(r_k) * cos2(psi_j) * cos2(psi_kl)
              + LLt(r_k) * sin2(psi_j) * sin2(psi_kl) )
            * ( LLp(r_jl) * cos2(psi_jl - psij) * cos2(psi_jl - psi_kl)
              + LLt(r_jl) * sin2(psi_jl - psi_j) * sin2(psi_jl - psi_kl) )

        f *= 2 * np.pi * r_k
        
        return f
    
    def integrand_pt(params):
        
        psi_j, psi_kl, r_j, r_kl, r_k = params
    
        y_kj = r_j*np.sin(psi_j)
        x_kj = r_j*np.cos(psi_j) - r_k
        
        r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
        psi_kj = np.arctan2(y_kj, x_kj)
        
        r_jl = cos_law_side(r_kl, r_kj, (psi_kl-psi_kj))
        psi_jl = cos_law_angle(r_kl, r_jl, r_kj) + psi_kl

        f = - ( LLp(r_k) * cos2(psi_j) * sin2(psi_kl)
              - LLt(r_k) * sin2(psi_j) * cos2(psi_kl) )
            * ( LLp(r_jl) * cos2(psi_jl - psij) * sin2(psi_jl - psi_kl)
              - LLt(r_jl) * sin2(psi_jl - psi_j) * cos2(psi_jl - psi_kl) )

        f *= 2 * np.pi * r_k
        
        return f
    
    def integrand_tt(params):
        
        psi_j, psi_kl, r_j, r_kl, r_k = params
    
        y_kj = r_j*np.sin(psi_j)
        x_kj = r_j*np.cos(psi_j) - r_k
        
        r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
        psi_kj = np.arctan2(y_kj, x_kj)
        
        r_jl = cos_law_side(r_kl, r_kj, (psi_kl-psi_kj))
        psi_jl = cos_law_angle(r_kl, r_jl, r_kj) + psi_kl

        f = ( LLp(r_k) * sin2(psi_j) * sin2(psi_kl)
              + LLt(r_k) * cos2(psi_j) * cos2(psi_kl) )
            * ( LLp(r_jl) * sin2(psi_jl - psij) * sin2(psi_jl - psi_kl)
              + LLt(r_jl) * cos2(psi_jl - psi_j) * cos2(psi_jl - psi_kl) )

        f *= 2 * np.pi * r_k
        
        return f
    
    def integral_bins(integrand, alpha, beta, nsampl):
        
        ranges = [(0, 2*np.pi), (0, 2*np.pi),
                  (rs[alpha], rs[alpha+1]), (rs[beta], rs[beta+1]), (0, r2_max)]
        
        integral, err = monte_carlo_integrate(integrand, ranges, nsampl)
        
        # normalisation of differential elements
        integral *= 2/(Omegatot * Omegas[alpha] * Omegas[beta]) 
        err *= 2/(Omegatot * Omegas[alpha] * Omegas[beta]) 
        return integral, err
    
    for alpha in range(Nbin1):
        for beta in range(Nbin2): 
                     
            ccov_pp[alpha, beta], err_pp = integral_bins(integrand_pp, alpha, beta, Csamp)
            ccov_pt[alpha, beta], err_pt = integral_bins(integrand_pt, alpha, beta, Csamp)
            ccov_tt[alpha, beta], err_tt = integral_bins(integrand_tt, alpha, beta, Csamp)

            test_err(err_pp, ccov_pp[alpha, beta], f'LLLL ccov plus plus angular bins {alpha, beta}')
            test_err(err_pt, ccov_pt[alpha, beta], f'LLLL ccov plus times angular bins {alpha, beta}')
            test_err(err_tt, ccov_tt[alpha, beta], f'LLLL ccov times times angular bins {alpha, beta}')
            
    # Make the full cosmic covariance matrix (need to fix this to make plus and minus)

    ccov = np.block([[ccov_p],
                     [ccov_t]])
    
    return ccov

################################################## 6.2.2 LeLe noise/sparsity covariance #############################################################

def generate_ncov_LLLL(delta_BD, sigma_LOS, distributions, approx=False):
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
    
    def integrand_pp(params):
        
        r_j, r_l, psi_l = params
    
        y_jl = r_l*np.sin(psi_l)
        x_jl = r_l*np.cos(psi_l) - r_j
        
        r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)
        
        f = cos2(psi_l) 
            * ( LLp(r_jl) * cos2(psi_jl) * cos2(psi_jl - psi_l) + LLt(psi_jl) * sin2(psi_jl) * sin2(psi_jl-psi_l) )

        f *= 2 * np.pi * r_j
                              
        return f
    
    def integrand_pt(params):
        
        r_j, r_l, psi_l = params
    
        y_jl = r_l*np.sin(psi_l)
        x_jl = r_l*np.cos(psi_l) - r_j
        
        r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)
        
        f = - sin2(psi_l) 
            * ( LLp(r_jl) * cos2(psi_jl) * sin2(psi_jl - psi_l) - LLt(psi_jl) * sin2(psi_jl) * cos2(psi_jl-psi_l) )

        f *= 2 * np.pi * r_j
                              
        return f
    
    def integrand_tt(params):
        
        r_j, r_l, psi_l = params
    
        y_jl = r_l*np.sin(psi_l)
        x_jl = r_l*np.cos(psi_l) - r_j
        
        r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)
        
        f = cos2(psi_l) 
            * ( LLp(r_jl) * sin2(psi_jl) * sin2(psi_jl - psi_l) + LLt(psi_jl) * cos2(psi_jl) * cos2(psi_jl-psi_l) )

        f *= 2 * np.pi * r_j
                              
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
                     
            ncov_pp[alpha, beta], err_pp = integral_bins(integrand_pp, alpha, beta, Csamp)
            ncov_pt[alpha, beta], err_pt = integral_bins(integrand_pt, alpha, beta, Csamp)
            ncov_tt[alpha, beta], err_tt = integral_bins(integrand_tt, alpha, beta, Csamp)

            test_err(err_pp, ncov_pp[alpha, beta], f'LLLL ncov plus plus angular bins {alpha, beta}')
            test_err(err_pt, ncov_pt[alpha, beta], f'LLLL ncov plus times angular bins {alpha, beta}')
            test_err(err_tt, ncov_tt[alpha, beta], f'LLLL ncov times times angular bins {alpha, beta}')
            
    # Make the full cosmic covariance matrix (need to fix this to make plus and minus)

    ncov = np.block([[ncov_p],
                     [ncov_t]])
    
    return ncov