import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp', 'LLx', 'angular_distributions', 'L0')

################################################## LLLL cosmic covariance ##############################################################

def generate_ccov_LLLL():
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear correlation functions.
    """

    def generate_matrices(sign1, sign2):   
    
        angular_distribution = angular_distributions[f'LL_{sign1}']
        angular_distribution = angular_distributions[f'LL_{sign2}']
        
        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL 
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad)
        
        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LL 
        Omegas2     = angular_distribution2.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LL (in rad)
        
        # Initialise the blocks
        ccov_pp = np.zeros((Nbin1, Nbin2))
        ccov_px = np.zeros((Nbin1, Nbin2))
        ccov_xp = np.zeros((Nbin1, Nbin2))
        ccov_xx = np.zeros((Nbin1, Nbin2))
        
        # Define the integrands (complete from here)
        
        def integrand_pp(params):
            
            psi_j, psi_kl, r_j, r_kl, r_k = params
        
            y_kj = r_j*np.sin(psi_j)
            x_kj = r_j*np.cos(psi_j) - r_k
            
            r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
            psi_kj = np.arctan2(y_kj, x_kj)
            
            r_jl = cos_law_side(r_kl, r_kj, (psi_kl-psi_kj))
            psi_jl = cos_law_angle(r_kl, r_jl, r_kj) + psi_kl
    
            f = ( ( LLp(r_k) * cos2(psi_j) * cos2(psi_kl)
                  + LLx(r_k) * sin2(psi_j) * sin2(psi_kl) )
            * ( LLp(r_jl) * cos2(psi_jl - psi_j) * cos2(psi_jl - psi_kl)
                  + LLx(r_jl) * sin2(psi_jl - psi_j) * sin2(psi_jl - psi_kl) ) )
    
            f *= 2 * np.pi * r_k * r_j * r_kl
            
            return f
        
        def integrand_px(params):
            
            psi_j, psi_kl, r_j, r_kl, r_k = params
        
            y_kj = r_j*np.sin(psi_j)
            x_kj = r_j*np.cos(psi_j) - r_k
            
            r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
            psi_kj = np.arctan2(y_kj, x_kj)
            
            r_jl = cos_law_side(r_kl, r_kj, (psi_kl-psi_kj))
            psi_jl = cos_law_angle(r_kl, r_jl, r_kj) + psi_kl
    
            f = - ( ( LLp(r_k) * cos2(psi_j) * sin2(psi_kl)
                  - LLx(r_k) * sin2(psi_j) * cos2(psi_kl) ) * ( LLp(r_jl) * cos2(psi_jl - psi_j) * sin2(psi_jl - psi_kl)
                  - LLx(r_jl) * sin2(psi_jl - psi_j) * cos2(psi_jl - psi_kl) ) )
    
            f *= 2 * np.pi * r_k * r_j * r_kl
            
            return f
        
        def integrand_xp(params):
            
            psi_j, psi_kl, r_j, r_kl, r_k = params
        
            y_kj = r_j*np.sin(psi_j)
            x_kj = r_j*np.cos(psi_j) - r_k
            
            r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
            psi_kj = np.arctan2(y_kj, x_kj)
            
            r_jl = cos_law_side(r_kl, r_kj, (psi_kl-psi_kj))
            psi_jl = cos_law_angle(r_kl, r_jl, r_kj) + psi_kl
    
            f = - ( ( LLp(r_k) * sin2(psi_j) * cos2(psi_kl)
                  - LLx(r_k) * cos2(psi_j) * sin2(psi_kl) ) * ( LLp(r_jl) * sin2(psi_jl - psi_j) * cos2(psi_jl - psi_kl)
                  - LLx(r_jl) * cos2(psi_jl - psi_j) * sin2(psi_jl - psi_kl) ) )
    
            f *= 2 * np.pi * r_k * r_j * r_kl
            
            return f
        
        def integrand_xx(params):
            
            psi_j, psi_kl, r_j, r_kl, r_k = params
        
            y_kj = r_j*np.sin(psi_j)
            x_kj = r_j*np.cos(psi_j) - r_k
            
            r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
            psi_kj = np.arctan2(y_kj, x_kj)
            
            r_jl = cos_law_side(r_kl, r_kj, (psi_kl-psi_kj))
            psi_jl = cos_law_angle(r_kl, r_jl, r_kj) + psi_kl
    
            f = ( ( LLp(r_k) * sin2(psi_j) * sin2(psi_kl)
                  + LLx(r_k) * cos2(psi_j) * cos2(psi_kl) ) * ( LLp(r_jl) * sin2(psi_jl - psi_j) * sin2(psi_jl - psi_kl)
                  + LLx(r_jl) * cos2(psi_jl - psi_j) * cos2(psi_jl - psi_kl) ) )
    
            f *= 2 * np.pi * r_k * r_j * r_kl
            
            return f
        
        def integral_bins(integrand, alpha, beta):
            
            ranges = [(0, 2*np.pi), (0, 2*np.pi),
                      (rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, r2_max)]
            
            integral, err = monte_carlo_integrate(integrand, ranges, Csamp)
            
            # normalisation of differential elements
            integral *= 2/(Omegatot * Omegas1[alpha] * Omegas2[beta]) 
            err *= 2/(Omegatot * Omegas1[alpha] * Omegas2[beta])
            
            return integral, err
        
        for alpha in range(Nbin1):
            for beta in range(Nbin2): 
                         
                ccov_pp[alpha, beta], err_pp = integral_bins(integrand_pp, alpha, beta)
                ccov_px[alpha, beta], err_px = integral_bins(integrand_px, alpha, beta)
                ccov_xp[alpha, beta], err_xp = integral_bins(integrand_xp, alpha, beta)
                ccov_xx[alpha, beta], err_xx = integral_bins(integrand_xx, alpha, beta)
    
                test_err(err_pp, ccov_pp[alpha, beta], f'LLLL ccov plus plus angular bins {alpha, beta}')
                test_err(err_px, ccov_px[alpha, beta], f'LLLL ccov plus times angular bins {alpha, beta}')
                test_err(err_xp, ccov_xp[alpha, beta], f'LLLL ccov times plus angular bins {alpha, beta}')
                test_err(err_xx, ccov_xx[alpha, beta], f'LLLL ccov times times angular bins {alpha, beta}')

        return ccov_pp, ccov_px, ccov_xp, ccov_xx

    #ccov_pp

    ccov_pp, ccov_px, ccov_xp, ccov_xx = generate_matrices('plus', 'plus')
    
    ccovpp = ccov_pp + ccov_px + ccov_xp + ccov_xx

    #ccov_pm

    ccov_pp, ccov_px, ccov_xp, ccov_xx = generate_matrices('plus', 'minus')
    
    ccovpm = ccov_pp - ccov_px + ccov_xp - ccov_xx

    #ccov_mp

    ccov_pp, ccov_px, ccov_xp, ccov_xx = generate_matrices('minus', 'plus')

    ccovmp = ccov_pp + ccov_px - ccov_xp - ccov_xx

    #ccov_mm

    ccov_pp, ccov_px, ccov_xp, ccov_xx = generate_matrices('minus', 'minus')
    
    ccovmm = ccov_pp - ccov_px - ccov_xp + ccov_xx
    
    ccov = np.block([[ccovmm, ccovmp],
                     [ccovpm, ccovpp]])
    
    return ccov

################################################## 6.2.2 LeLe noise/sparsity covariance #############################################################

def generate_ncov_LLLL():
    """
    Computes the contribution of noise and sparsity variance in the 
    covariance matrix of the LOS shear correlation functions.
    """

    def generate_matrices(sign1, sign2):  
    
        angular_distribution1 = angular_distributions[f'LL_{sign1}']
        angular_distribution2 = angular_distributions[f'LL_{sign2}']
        
        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL (sign1) 
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (sign1)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad) (sign1)
        
        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LL (sign2)
        Omegas2     = angular_distribution2.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (sign2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LL (in rad) (sign2)
        
        # Initialise the blocks
        ncov_pp = np.zeros((Nbin1, Nbin2))
        ncov_px = np.zeros((Nbin1, Nbin2))
        ncov_xp = np.zeros((Nbin1, Nbin2))
        ncov_xx = np.zeros((Nbin1, Nbin2))
        
        # Initialise the blocks
        scov_pp = np.zeros((Nbin1, Nbin2))
        scov_px = np.zeros((Nbin1, Nbin2))
        scov_xp = np.zeros((Nbin1, Nbin2))
        scov_xx = np.zeros((Nbin1, Nbin2))
        
        # Define the integrands
        
        def integrand_pp(params):
            
            r_j, r_l, psi_l = params
        
            y_jl = r_l*np.sin(psi_l)
            x_jl = r_l*np.cos(psi_l) - r_j
            
            r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
            psi_jl = np.arctan2(y_jl, x_jl)
            
            f = cos2(psi_l) * ( LLp(r_jl) * cos2(psi_jl) * cos2(psi_jl - psi_l) + LLx(r_jl) * sin2(psi_jl) * sin2(psi_jl-psi_l) )
    
            f *= 2 * np.pi * r_j * r_l
                                  
            return f
        
        def integrand_px(params):
            
            r_j, r_l, psi_l = params
        
            y_jl = r_l*np.sin(psi_l)
            x_jl = r_l*np.cos(psi_l) - r_j
            
            r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
            psi_jl = np.arctan2(y_jl, x_jl)
            
            f = - sin2(psi_l) * ( LLp(r_jl) * cos2(psi_jl) * sin2(psi_jl - psi_l) - LLx(r_jl) * sin2(psi_jl) * cos2(psi_jl-psi_l) )
    
            f *= 2 * np.pi * r_j * r_l
                                  
            return f
        
        def integrand_xp(params):
            
            r_j, r_l, psi_l = params
        
            y_jl = r_l*np.sin(psi_l)
            x_jl = r_l*np.cos(psi_l) - r_j
            
            r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
            psi_jl = np.arctan2(y_jl, x_jl)
            
            f = sin2(psi_l) * ( LLp(r_jl) * sin2(psi_jl) * cos2(psi_jl - psi_l) - LLx(r_jl) * cos2(psi_jl) * sin2(psi_jl-psi_l) )
    
            f *= 2 * np.pi * r_j * r_l
                                  
            return f
        
        def integrand_xx(params):
            
            r_j, r_l, psi_l = params
        
            y_jl = r_l*np.sin(psi_l)
            x_jl = r_l*np.cos(psi_l) - r_j
            
            r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
            psi_jl = np.arctan2(y_jl, x_jl)
            
            f = cos2(psi_l) * ( LLp(r_jl) * sin2(psi_jl) * sin2(psi_jl - psi_l) + LLx(r_jl) * cos2(psi_jl) * cos2(psi_jl-psi_l) )
    
            f *= 2 * np.pi * r_j * r_l
                                  
            return f
        
        def integral_bins(integrand, alpha, beta):
            
            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]
            
            integral, err = monte_carlo_integrate(integrand, ranges, Nsamp)
            
            # normalisation of differential elements
            integral *= 2/(Omegas1[alpha] * Omegas2[beta]) 
            err      *= 2/(Omegas1[alpha] * Omegas2[beta]) 
            
            return integral, err
        
        for alpha in range(Nbin1):
            for beta in range(Nbin2): 
                         
                int_pp, err_pp = integral_bins(integrand_pp, alpha, beta)
                int_px, err_px = integral_bins(integrand_px, alpha, beta)
                int_xp, err_xp = integral_bins(integrand_xp, alpha, beta)
                int_xx, err_xx = integral_bins(integrand_xx, alpha, beta)
    
                if alpha == beta:
                    if alpha == 0:
                        print(int_pp)
                         
                ncov_pp[alpha, beta] = (sigma_L**2/Nlens) * int_pp
                nerr_pp = (sigma_L**2/Nlens) * err_pp
                         
                scov_pp[alpha, beta] = (L0/Nlens) * int_pp
                serr_pp = (L0/Nlens) * err_pp
                         
                ncov_px[alpha, beta] = (sigma_L**2/Nlens) * int_px
                nerr_px = (sigma_L**2/Nlens) * err_px
                         
                scov_px[alpha, beta] = (L0/Nlens) * int_px
                serr_px = (L0/Nlens) * err_px
                         
                ncov_xp[alpha, beta] = (sigma_L**2/Nlens) * int_xp
                nerr_xp = (sigma_L**2/Nlens) * err_xp
                         
                scov_xp[alpha, beta] = (L0/Nlens) * int_xp
                serr_xp = (L0/Nlens) * err_xp
                         
                ncov_xx[alpha, beta] = (sigma_L**2/Nlens) * int_xx
                nerr_xx = (sigma_L**2/Nlens) * err_xx
                         
                scov_xx[alpha, beta] = (L0/Nlens) * int_xx
                serr_xx = (L0/Nlens) * err_xx
        
                #addition of constant term
                if alpha == beta:
                    ncov_pp[alpha, beta] += (1/2) * ( (sigma_L**2/Nlens)**2 ) * (Omegatot/Omegas[alpha])
                    ncov_xx[alpha, beta] += (1/2) * ( (sigma_L**2/Nlens)**2 ) * (Omegatot/Omegas[alpha])
                    
                    scov_pp[alpha, beta] += (1/2) * ( (L0/Nlens)**2 ) * (Omegatot/Omegas[alpha])
                    scov_xx[alpha, beta] += (1/2) * ( (L0/Nlens)**2 ) * (Omegatot/Omegas[alpha])
    
                test_err(nerr_pp, ncov_pp[alpha, beta], f'LLLL ncov plus plus angular bins {alpha, beta}')
                test_err(nerr_px, ncov_px[alpha, beta], f'LLLL ncov plus times angular bins {alpha, beta}')
                test_err(nerr_xp, ncov_xp[alpha, beta], f'LLLL ncov times plus angular bins {alpha, beta}')
                test_err(nerr_xx, ncov_xx[alpha, beta], f'LLLL ncov times times angular bins {alpha, beta}')
    
                test_err(serr_pp, scov_pp[alpha, beta], f'LLLL scov plus plus angular bins {alpha, beta}')
                test_err(serr_px, scov_px[alpha, beta], f'LLLL scov plus times angular bins {alpha, beta}')
                test_err(serr_xp, scov_xp[alpha, beta], f'LLLL scov times plus angular bins {alpha, beta}')
                test_err(serr_xx, scov_xx[alpha, beta], f'LLLL scov times times angular bins {alpha, beta}')

        return ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx    
    
    #plus plus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx = generate_matrices('plus', 'plus')
    
    ncovpp = ncov_pp + ncov_px + ncov_xp + ncov_xx    
    scovpp = scov_pp + scov_px + scov_xp + scov_xx

    #plus minus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx = generate_matrices('plus', 'minus')
    
    ncovpm = ncov_pp - ncov_px + ncov_xp - ncov_xx
    scovpm = scov_pp - scov_px + scov_xp - scov_xx

    #minus plus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx = generate_matrices('minus', 'plus')
    
    ncovmp = ncov_pp + ncov_px - ncov_xp - ncov_xx
    scovmp = scov_pp + scov_px - scov_xp - scov_xx

    #minus minus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx = generate_matrices('minus', 'minus')
    
    ncovmm = ncov_pp - ncov_px - ncov_xp + ncov_xx
    scovmm = scov_pp - scov_px - scov_xp + scov_xx
    
    ncov = np.block([[ncovmm, ncovmp],
                     [ncovpm, ncovpp]])
    
    scov = np.block([[scovmm, scovmp],
                     [scovpm, scovpp]])
    
    return [ncov, scov]