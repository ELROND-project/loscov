import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp','LLx', 'LEp', 'LEx', 'angular_distributions', 'redshift_distributions', 'L0')

################################################## LLLE cosmic covariance ##############################################################

def generate_ccov_LLLE(D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - LOS shear x LOS shear - galaxy shape correlation functions.

    D             : the galaxy redshift bin D (0 to Nbinz_E)
    """

    def generate_matrices(sign1, sign2): 
    
        angular_distribution1 = angular_distributions[f'LL_{sign1}']
        angular_distribution2 = angular_distributions[f'LE_{sign2}'][D]
        
        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL 
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad)
        
        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LE 
        Omegas2     = angular_distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LE (in rad)
        
        # Initialise the blocks
        ccov_pp = np.zeros((Nbin1, Nbin2))
        ccov_px = np.zeros((Nbin1, Nbin2))
        ccov_xp = np.zeros((Nbin1, Nbin2))
        ccov_xx = np.zeros((Nbin1, Nbin2))
        
        err_pp = np.zeros((Nbin1, Nbin2))
        err_px = np.zeros((Nbin1, Nbin2))
        err_xp = np.zeros((Nbin1, Nbin2))
        err_xx = np.zeros((Nbin1, Nbin2))    
        
        # Define the integrands (compLExe from here)
        
        def integrand_pp(params):
            
            psi_j, psi_kd, r_j, r_kd, r_k = params
        
            y_kj = r_j*np.sin(psi_j)
            x_kj = r_j*np.cos(psi_j) - r_k
            
            r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
            psi_kj = np.arctan2(y_kj, x_kj)
            
            r_jd = cos_law_side(r_kd, r_kj, (psi_kd-psi_kj))
            psi_jd = cos_law_angle(r_kd, r_jd, r_kj) + psi_kd
    
            f = ( ( LLp(r_k) * cos2(psi_j) * cos2(psi_kd)
                  + LLx(r_k) * sin2(psi_j) * sin2(psi_kd) )
                * ( LEp[D](r_jd) * cos2(psi_jd - psi_j) * cos2(psi_jd - psi_kd)
                  + LEx[D](r_jd) * sin2(psi_jd - psi_j) * sin2(psi_jd - psi_kd) )
                )
            
            f *= 2 * np.pi * r_k * r_j * r_kd
            
            return f
        
        def integrand_px(params):
            
            psi_j, psi_kd, r_j, r_kd, r_k = params
        
            y_kj = r_j*np.sin(psi_j)
            x_kj = r_j*np.cos(psi_j) - r_k
            
            r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
            psi_kj = np.arctan2(y_kj, x_kj)
            
            r_jd = cos_law_side(r_kd, r_kj, (psi_kd-psi_kj))
            psi_jd = cos_law_angle(r_kd, r_jd, r_kj) + psi_kd
    
            f = -( ( LLp(r_k) * cos2(psi_j) * sin2(psi_kd)
                  - LLx(r_k) * sin2(psi_j) * cos2(psi_kd) )
                * ( LEp[D](r_jd) * cos2(psi_jd - psi_j) * sin2(psi_jd - psi_kd)
                  - LEx[D](r_jd) * sin2(psi_jd - psi_j) * cos2(psi_jd - psi_kd) )
                )
            
            f *= 2 * np.pi * r_k * r_j * r_kd
            
            return f
        
        def integrand_xp(params):
            
            psi_j, psi_kd, r_j, r_kd, r_k = params
        
            y_kj = r_j*np.sin(psi_j)
            x_kj = r_j*np.cos(psi_j) - r_k
            
            r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
            psi_kj = np.arctan2(y_kj, x_kj)
            
            r_jd = cos_law_side(r_kd, r_kj, (psi_kd-psi_kj))
            psi_jd = cos_law_angle(r_kd, r_jd, r_kj) + psi_kd
    
            f = -( ( LLp(r_k) * sin2(psi_j) * cos2(psi_kd)
                  - LLx(r_k) * cos2(psi_j) * sin2(psi_kd) )
                * ( LEp[D](r_jd) * sin2(psi_jd - psi_j) * cos2(psi_jd - psi_kd)
                  - LEx[D](r_jd) * cos2(psi_jd - psi_j) * sin2(psi_jd - psi_kd) )
                )
            
            f *= 2 * np.pi * r_k * r_j * r_kd
            
            return f
        
        def integrand_xx(params):
            
            psi_j, psi_kd, r_j, r_kd, r_k = params
        
            y_kj = r_j*np.sin(psi_j)
            x_kj = r_j*np.cos(psi_j) - r_k
            
            r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
            psi_kj = np.arctan2(y_kj, x_kj)
            
            r_jd = cos_law_side(r_kd, r_kj, (psi_kd-psi_kj))
            psi_jd = cos_law_angle(r_kd, r_jd, r_kj) + psi_kd
    
            f = ( ( LLp(r_k) * sin2(psi_j) * sin2(psi_kd)
                  + LLx(r_k) * cos2(psi_j) * cos2(psi_kd) )
                * ( LEp[D](r_jd) * sin2(psi_jd - psi_j) * sin2(psi_jd - psi_kd)
                  + LEx[D](r_jd) * cos2(psi_jd - psi_j) * cos2(psi_jd - psi_kd) )
                )
            
            f *= 2 * np.pi * r_k * r_j * r_kd
            
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
                         
                ccov_pp[alpha, beta], err_pp[alpha, beta] = integral_bins(integrand_pp, alpha, beta)
                ccov_px[alpha, beta], err_px[alpha, beta] = integral_bins(integrand_px, alpha, beta)
                ccov_xp[alpha, beta], err_xp[alpha, beta] = integral_bins(integrand_xp, alpha, beta)
                ccov_xx[alpha, beta], err_xx[alpha, beta] = integral_bins(integrand_xx, alpha, beta)
    
                test_err(err_pp[alpha, beta], ccov_pp[alpha, beta], f'LLLE ccov plus plus angular bins {alpha, beta}')
                test_err(err_px[alpha, beta], ccov_px[alpha, beta], f'LLLE ccov plus times angular bins {alpha, beta}')
                test_err(err_xp[alpha, beta], ccov_xp[alpha, beta], f'LLLE ccov times plus angular bins {alpha, beta}')
                test_err(err_xx[alpha, beta], ccov_xx[alpha, beta], f'LLLE ccov times times angular bins {alpha, beta}')

        err = np.sqrt(err_pp**2 + err_px**2 + err_xp**2 + err_xx**2)

        return ccov_pp, ccov_px, ccov_xp, ccov_xx, err

    #plus plus

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errpp = generate_matrices('plus', 'plus')
    ccovpp = ccov_pp + ccov_px + ccov_xp + ccov_xx
    
    #plus minus

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errpm = generate_matrices('plus', 'minus')
    ccovpm = ccov_pp - ccov_px + ccov_xp - ccov_xx

    #minus plus

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errmp = generate_matrices('minus', 'plus')
    ccovmp = ccov_pp + ccov_px - ccov_xp - ccov_xx

    #minus minus

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errmm = generate_matrices('minus', 'minus')
    ccovmm = ccov_pp - ccov_px - ccov_xp + ccov_xx
    
    ccov = np.block([[ccovmm, ccovmp],
                     [ccovpm, ccovpp]])
    
    err = np.block([[errmm, errmp],
                     [errpm, errpp]])
    
    return ccov, err

################################################## LLLE noise/sparsity covariance #############################################################

def generate_ncov_LLLE(D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear correlation functions.

    D             : the galaxy redshift bin D (0 to Nbinz_E)
    """

    def generate_matrices(sign1, sign2): 
    
        angular_distribution1 = angular_distributions[f'LL_{sign1}']
        angular_distribution2 = angular_distributions[f'LE_{sign2}'][D]
        
        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL 
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad)
        
        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LE 
        Omegas2     = angular_distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LE (in rad)
        
        # Initialise the blocks
        ncov_pp = np.zeros((Nbin1, Nbin2))
        ncov_px = np.zeros((Nbin1, Nbin2))
        ncov_xp = np.zeros((Nbin1, Nbin2))
        ncov_xx = np.zeros((Nbin1, Nbin2))
        
        nerr_pp = np.zeros((Nbin1, Nbin2))
        nerr_px = np.zeros((Nbin1, Nbin2))
        nerr_xp = np.zeros((Nbin1, Nbin2))
        nerr_xx = np.zeros((Nbin1, Nbin2))
        
        scov_pp = np.zeros((Nbin1, Nbin2))
        scov_px = np.zeros((Nbin1, Nbin2))
        scov_xp = np.zeros((Nbin1, Nbin2))
        scov_xx = np.zeros((Nbin1, Nbin2))
        
        serr_pp = np.zeros((Nbin1, Nbin2))
        serr_px = np.zeros((Nbin1, Nbin2))
        serr_xp = np.zeros((Nbin1, Nbin2))
        serr_xx = np.zeros((Nbin1, Nbin2))
        
        # Define the integrands
        
        def integrand_pp(params):
            
            r_j, r_d, psi_d = params
        
            y_jd = r_d*np.sin(psi_d)
            x_jd = r_d*np.cos(psi_d) - r_j
            
            r_jd = np.sqrt( y_jd**2 + x_jd**2 ) 
            psi_jd = np.arctan2(y_jd, x_jd)
            
            f = ( cos2(psi_d) 
                * ( LEp[D](r_jd) * cos2(psi_jd) * cos2(psi_jd - psi_d) + LEx[D](r_jd) * sin2(psi_jd) * sin2(psi_jd-psi_d) ) )
            
            f *= 2 * np.pi * r_j * r_d
                                  
            return f
        
        def integrand_px(params):
            
            r_j, r_d, psi_d = params
        
            y_jd = r_d*np.sin(psi_d)
            x_jd = r_d*np.cos(psi_d) - r_j
            
            r_jd = np.sqrt( y_jd**2 + x_jd**2 ) 
            psi_jd = np.arctan2(y_jd, x_jd)
            
            f = - ( sin2(psi_d) 
                * ( LEp[D](r_jd) * cos2(psi_jd) * sin2(psi_jd - psi_d) - LEx[D](r_jd) * sin2(psi_jd) * cos2(psi_jd-psi_d) ) )
    
            f *= 2 * np.pi * r_j * r_d
                                  
            return f
        
        def integrand_xp(params):
            
            r_j, r_d, psi_d = params
        
            y_jd = r_d*np.sin(psi_d)
            x_jd = r_d*np.cos(psi_d) - r_j
            
            r_jd = np.sqrt( y_jd**2 + x_jd**2 ) 
            psi_jd = np.arctan2(y_jd, x_jd)
            
            f = ( sin2(psi_d) 
                * ( LEp[D](r_jd) * sin2(psi_jd) * cos2(psi_jd - psi_d) - LEx[D](r_jd) * cos2(psi_jd) * sin2(psi_jd-psi_d) ) )
    
            f *= 2 * np.pi * r_j * r_d
                                  
            return f
        
        def integrand_xx(params):
            
            r_j, r_d, psi_d = params
        
            y_jd = r_d*np.sin(psi_d)
            x_jd = r_d*np.cos(psi_d) - r_j
            
            r_jd = np.sqrt( y_jd**2 + x_jd**2 ) 
            psi_jd = np.arctan2(y_jd, x_jd)
            
            f = ( cos2(psi_d) 
                * ( LEp[D](r_jd) * sin2(psi_jd) * sin2(psi_jd - psi_d) + LEx[D](r_jd) * cos2(psi_jd) * cos2(psi_jd-psi_d) ) )
    
            f *= 2 * np.pi * r_j * r_d
                                  
            return f
        
        def integral_bins(integrand, alpha, beta):
            
            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]
            
            integral, err = monte_carlo_integrate(integrand, ranges, Nsamp)
            
            # normalisation of differential elements
            integral /= (Omegas1[alpha] * Omegas2[beta]) 
            err      /= (Omegas1[alpha] * Omegas2[beta]) 
            
            return integral, err
        
        for alpha in range(Nbin1):
            for beta in range(Nbin2): 
                         
                int_pp, err_pp = integral_bins(integrand_pp, alpha, beta)
                int_px, err_px = integral_bins(integrand_px, alpha, beta)
                int_xp, err_xp = integral_bins(integrand_xp, alpha, beta)
                int_xx, err_xx = integral_bins(integrand_xx, alpha, beta)
                         
                ncov_pp[alpha, beta] = (sigma_L**2/Nlens) * int_pp
                nerr_pp[alpha, beta] = (sigma_L**2/Nlens) * err_pp
                         
                scov_pp[alpha, beta] = (L0/Nlens) * int_pp
                serr_pp[alpha, beta] = (L0/Nlens) * err_pp
                         
                ncov_px[alpha, beta] = (sigma_L**2/Nlens) * int_px
                nerr_px[alpha, beta] = (sigma_L**2/Nlens) * err_px
                         
                scov_px[alpha, beta] = (L0/Nlens) * int_px
                serr_px[alpha, beta] = (L0/Nlens) * err_px
                         
                ncov_xp[alpha, beta] = (sigma_L**2/Nlens) * int_xp
                nerr_xp[alpha, beta] = (sigma_L**2/Nlens) * err_xp
                         
                scov_xp[alpha, beta] = (L0/Nlens) * int_xp
                serr_xp[alpha, beta] = (L0/Nlens) * err_xp
                         
                ncov_xx[alpha, beta] = (sigma_L**2/Nlens) * int_xx
                nerr_xx[alpha, beta] = (sigma_L**2/Nlens) * err_xx
                         
                scov_xx[alpha, beta] = (L0/Nlens) * int_xx
                serr_xx[alpha, beta] = (L0/Nlens) * err_xx
    
                test_err(nerr_pp[alpha, beta], ncov_pp[alpha, beta], f'LLLE ncov plus plus angular bins {alpha, beta}')
                test_err(nerr_px[alpha, beta], ncov_px[alpha, beta], f'LLLE ncov plus times angular bins {alpha, beta}')
                test_err(nerr_xp[alpha, beta], ncov_xp[alpha, beta], f'LLLE ncov times plus angular bins {alpha, beta}')
                test_err(nerr_xx[alpha, beta], ncov_xx[alpha, beta], f'LLLE ncov times times angular bins {alpha, beta}')
    
                test_err(serr_pp[alpha, beta], scov_pp[alpha, beta], f'LLLE scov plus plus angular bins {alpha, beta}')
                test_err(serr_px[alpha, beta], scov_px[alpha, beta], f'LLLE scov plus times angular bins {alpha, beta}')
                test_err(serr_xp[alpha, beta], scov_xp[alpha, beta], f'LLLE scov times plus angular bins {alpha, beta}')
                test_err(serr_xx[alpha, beta], scov_xx[alpha, beta], f'LLLE scov times times angular bins {alpha, beta}')

        nerr = np.sqrt(nerr_pp**2 + nerr_px**2 + nerr_xp**2 + nerr_xx**2)
        serr = np.sqrt(serr_pp**2 + serr_px**2 + serr_xp**2 + serr_xx**2) 

        return ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerr, serr 
    
    #plus plus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerrpp, serrpp  = generate_matrices('plus', 'plus')
    
    ncovpp = ncov_pp + ncov_px + ncov_xp + ncov_xx
    scovpp = scov_pp + scov_px + scov_xp + scov_xx
    
    #plus minus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerrpm, serrpm = generate_matrices('plus', 'minus')
    
    ncovpm = ncov_pp - ncov_px + ncov_xp - ncov_xx
    scovpm = scov_pp - scov_px + scov_xp - scov_xx
    
    #minus plus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerrmp, serrmp = generate_matrices('minus', 'plus')
    
    ncovmp = ncov_pp + ncov_px - ncov_xp - ncov_xx
    scovmp = scov_pp + scov_px - scov_xp - scov_xx
    
    #minus minus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerrmm, serrmm = generate_matrices('minus', 'minus')
    
    ncovmm = ncov_pp - ncov_px - ncov_xp + ncov_xx
    scovmm = scov_pp - scov_px - scov_xp + scov_xx

    ncov = np.block([[ncovmm, ncovmp],
                     [ncovpm, ncovpp]])

    nerr = np.block([[nerrmm, nerrmp],
                     [nerrpm, nerrpp]])
    
    scov = np.block([[scovmm, scovmp],
                     [scovpm, scovpp]])

    serr = np.block([[serrmm, serrmp],
                     [serrpm, serrpp]])
    
    return [ncov, scov], [nerr, serr]
